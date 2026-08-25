"""
Stripe integration playbook for Aria.

This is NOT loaded by default. server.py detects an explicit user request
for Stripe (keywords like "stripe", "plati cu cardul", "checkout", "subscription
billing") and appends STRIPE_PLAYBOOK to the builder system prompt only for
that turn — so ordinary app-building requests stay fast and don't carry this
extra weight.

The rules below exist because payment code has a different risk profile than
ordinary app code: a logic bug in a todo list loses a todo; a logic bug in
payment code loses real money, either the user's or the platform's. AI-
generated code that "looks right" is not sufficient here — these rules are
non-negotiable structural requirements, not style suggestions.
"""

STRIPE_PLAYBOOK = """
=== STRIPE INTEGRATION — MANDATORY RULES ===

You are now implementing payment functionality. The rules in this section
override any instinct to take shortcuts for simplicity. Payment code that
"looks correct" but skips these rules WILL be exploited — this is not a
hypothetical risk, it is the default outcome of client-trusting payment code
shipped to the public internet.

1. THE CLIENT IS NEVER TRUSTED FOR MONEY
   The mobile/web client NEVER decides: the price, the amount charged, the
   currency, the product being purchased, or whether a purchase succeeded.
   The client only tells the server WHAT the user wants to buy (a product ID,
   never a price). The server looks up the real price from its own trusted
   source (a database or hardcoded server-side price table) and creates the
   Stripe PaymentIntent/Checkout Session with that server-side price — never
   with a price or amount sent by the client. If you find yourself writing
   `amount: req.body.amount` anywhere in server code, stop — that is the
   single most common real-world Stripe exploit (attacker intercepts the
   request and changes the amount to $0.01).

2. WEBHOOKS ARE THE SOURCE OF TRUTH, NOT THE CLIENT REDIRECT
   After a Stripe Checkout/PaymentIntent, the client will get redirected to a
   "success" URL or receive a success callback in the SDK. This success
   signal from the client is NEVER sufficient to grant the purchased
   item/credits/subscription. An attacker can navigate directly to your
   success URL without ever paying. The ONLY trusted confirmation is a
   Stripe webhook event (checkout.session.completed,
   payment_intent.succeeded, invoice.paid, etc.) received server-side and
   verified with the webhook signing secret (stripe.Webhook.construct_event
   with STRIPE_WEBHOOK_SECRET). Grant the purchase / add the credits / start
   the subscription ONLY inside the webhook handler, after signature
   verification succeeds. The client-side success page should show a
   "processing" or "thank you" state, not itself trigger the grant.

3. IDEMPOTENCY — EVERY WEBHOOK CAN ARRIVE MORE THAN ONCE
   Stripe explicitly documents that webhook events can be delivered more
   than once (retries, duplicates). Before granting anything in a webhook
   handler, check whether you have already processed that specific
   event.id (or the underlying payment_intent id / session id) — store
   processed event IDs in the database and skip if already seen. Without
   this, a single successful payment can grant credits multiple times if
   Stripe retries the webhook (which it does routinely).

4. IDEMPOTENCY KEYS ON CHARGE CREATION
   When creating a PaymentIntent or Charge from your server in response to a
   client action (e.g. "user tapped Pay"), pass an idempotency_key derived
   from something stable about that specific attempt (e.g. a server-generated
   order ID), so that a network retry or a double-tap from the user cannot
   create two separate charges for one purchase.

5. SECRET KEYS NEVER REACH THE CLIENT
   The Stripe SECRET key (sk_live_... / sk_test_...) and the webhook signing
   secret live ONLY in server environment variables, read via os.environ,
   never hardcoded, never sent in any API response, never included in
   client-side bundles or mobile app code. Only the PUBLISHABLE key
   (pk_live_... / pk_test_...) is safe to ship to the client, because it is
   designed to be public — it can only be used to create tokens, not to
   charge anything or read account data.

6. AMOUNT AND CURRENCY VALIDATION, SERVER-SIDE
   Validate that the resolved server-side price for a product is a positive
   integer number of the smallest currency unit (e.g. cents for USD) before
   creating any Stripe object. Never trust a stored price of 0 or negative to
   silently succeed — treat that as a configuration error and reject the
   purchase attempt with a clear server-side error, not a $0 charge.

7. REFUNDS AND DISPUTES NEED A REVOCATION PATH
   If the product grants ongoing access (subscription, unlockable feature),
   design the data model so that a webhook for
   charge.refunded / charge.dispute.created / customer.subscription.deleted
   can revoke that access server-side. Do not build a system where access,
   once granted, has no code path to be taken away.

8. TEST MODE VS LIVE MODE
   Never hardcode assumptions that mix test and live keys. Read which mode to
   use from environment configuration, and make sure webhook secrets match
   the mode (test webhook secret for test mode, live webhook secret for live
   mode) — mismatched secrets cause `construct_event` to fail signature
   verification, which is a common integration bug, not a security issue,
   but will silently break granting purchases if not tested end-to-end in
   test mode first.

=== VERIFIED PATTERN: SERVER-SIDE CHECKOUT SESSION (Python / FastAPI style) ===

# Server holds the ONLY trusted price table.
PRODUCT_PRICES_CENTS = {
    "credits_100": 499,   # $4.99, defined server-side, never trust client amount
    "credits_500": 1999,
}

@app.post("/api/create-checkout-session")
async def create_checkout_session(body: CheckoutIn, user=Depends(get_current_user)):
    if body.product_id not in PRODUCT_PRICES_CENTS:
        raise HTTPException(400, "Produs necunoscut")
    price_cents = PRODUCT_PRICES_CENTS[body.product_id]  # NEVER body.amount

    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {"name": body.product_id},
                "unit_amount": price_cents,
            },
            "quantity": 1,
        }],
        success_url=f"{FRONTEND_URL}/purchase-processing?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{FRONTEND_URL}/purchase-cancelled",
        client_reference_id=user["id"],
        metadata={"user_id": user["id"], "product_id": body.product_id},
        idempotency_key=f"checkout-{user['id']}-{body.product_id}-{int(time.time() // 60)}",
    )
    return {"checkout_url": session.url}


# The ONLY place a purchase is actually granted — after signature verification.
@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(400, "Invalid signature")

    # Idempotency: never process the same event twice.
    already_processed = await db.processed_stripe_events.find_one({"id": event["id"]})
    if already_processed:
        return {"ok": True}  # acknowledge, but do nothing — already handled
    await db.processed_stripe_events.insert_one({"id": event["id"], "processed_at": now_iso()})

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["metadata"]["user_id"]
        product_id = session["metadata"]["product_id"]
        await grant_purchase(user_id, product_id)  # the actual credit/unlock happens ONLY here

    return {"ok": True}
"""
