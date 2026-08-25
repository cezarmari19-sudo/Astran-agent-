"""
Google Play Billing integration playbook for Aria.

NOT loaded by default. server.py detects an explicit user request for
Google Play Billing / Android in-app purchases (keywords like "google play
billing", "in-app purchase android", "IAP google", "cumparaturi in aplicatie
android") and appends GOOGLE_BILLING_PLAYBOOK to the builder system prompt
only for that turn.

Google Play Billing has a fundamentally different trust model than Stripe:
the PURCHASE itself happens entirely on-device through the Play Store app,
which the developer does not control. The server's job is not to initiate
the charge (Google does that) — the server's job is to verify that a
purchase token presented by the client is real, unconsumed, and actually
paid for, before granting anything. Skipping server-side verification is
the single most common Play Billing exploit: modified/cracked APKs and
Frida-based tooling can fake a successful purchase callback on-device while
no real payment ever reached Google.
"""

GOOGLE_BILLING_PLAYBOOK = """
=== GOOGLE PLAY BILLING INTEGRATION — MANDATORY RULES ===

You are now implementing Android in-app purchases. The client-side Billing
Library callback saying a purchase "succeeded" is NEVER sufficient by
itself to grant anything — it must always be independently verified
server-side against Google's servers before the purchase is trusted.

1. THE ON-DEVICE SUCCESS CALLBACK IS NOT PROOF OF PAYMENT
   The Play Billing Library on the device will call your app's purchase
   listener with a Purchase object once the Play Store UI reports success.
   This callback can be spoofed by a modified APK, a rooted device with
   Frida/Xposed hooking the Billing Library, or a repackaged app — none of
   which require an attacker to actually pay Google anything. Treat this
   callback ONLY as a signal to send the purchase token to YOUR server for
   verification — never as a reason to grant credits/unlock content
   directly from client-side code.

2. SERVER-SIDE VERIFICATION VIA THE GOOGLE PLAY DEVELOPER API
   Your server must call the Google Play Developer API
   (purchases.products.get for one-time products, or
   purchases.subscriptions.get for subscriptions) using a service account
   with access to your Play Console app, passing the purchaseToken the
   client sent you. Only if Google's API confirms purchaseState == 0
   (purchased) and the product ID matches what you expect should you grant
   the item. This requires a Google Cloud service account JSON key, stored
   server-side only (as an environment variable or secret file, never
   shipped in the app or committed to a repo).

3. ACKNOWLEDGE AND CONSUME PURCHASES CORRECTLY
   Google requires purchases to be acknowledged (purchases.products.acknowledge)
   within 3 days or they are automatically refunded. For consumable products
   (things a user can buy repeatedly, like a currency top-up), you must also
   call purchases.products.consume after granting the item, so the user can
   purchase it again. For non-consumable products (permanent unlocks), you
   acknowledge but do not consume. Getting this wrong either causes
   automatic refunds (acknowledgement missed) or blocks repeat purchases
   (consumption missed) — implement whichever the product type requires.

4. IDEMPOTENCY ON THE PURCHASE TOKEN
   Store every purchaseToken your server has already granted, in a database
   table with a unique constraint on the token. Before granting anything,
   check whether that exact token was already processed — a client retry,
   a network hiccup causing a duplicate API call, or a user replaying an
   old request must never grant the same purchase twice.

5. SUBSCRIPTIONS NEED SERVER-SIDE STATE, NOT CLIENT-SIDE ASSUMPTIONS
   For subscriptions, do not assume "purchased once = access forever" on the
   client. Store subscription state (active, expired, cancelled, in grace
   period) server-side, refreshed via the Developer API or via Real-time
   Developer Notifications (RTDN, a Pub/Sub feed Google can push
   subscription lifecycle events to) if you want near-real-time expiry
   handling. Gate access to subscription features by checking the
   server-side stored state, not a flag set once on the device at purchase
   time.

6. PRODUCT IDS AND PRICES ARE CONFIGURED IN PLAY CONSOLE, NOT YOUR CODE
   Unlike Stripe, prices for Play Billing products are configured in the
   Google Play Console, not computed by your server. Your server's job is
   to know which product IDs exist and map a verified purchase's product ID
   to what to grant — never to compute or trust a price value from the
   client at all, since Play Billing does not send one.

7. REVOCATION PATH FOR REFUNDS
   Google can refund a purchase (support request, chargeback) independent
   of your app. If you want to revoke access on refund, you need to poll
   purchases.products.get / purchases.subscriptions.get periodically, or
   subscribe to Real-time Developer Notifications, and check for a
   voided/refunded state — a purchase verified once at grant-time does not
   stay verified forever without ongoing checks for subscriptions and
   refundable consumables.

=== VERIFIED PATTERN: SERVER-SIDE PURCHASE VERIFICATION (Python / FastAPI style) ===

# Server holds the mapping from Play Console product ID to what it grants.
PRODUCT_GRANTS = {
    "credits_100": {"type": "consumable", "credits": 100},
    "premium_unlock": {"type": "non_consumable", "unlocks": "premium"},
}

@app.post("/api/verify-purchase")
async def verify_purchase(body: VerifyPurchaseIn, user=Depends(get_current_user)):
    if body.product_id not in PRODUCT_GRANTS:
        raise HTTPException(400, "Produs necunoscut")

    # Idempotency: never grant the same purchase token twice.
    already = await db.processed_purchase_tokens.find_one({"token": body.purchase_token})
    if already:
        return {"ok": True, "already_processed": True}

    # Verify against Google's servers — never trust the client's claim alone.
    android_publisher = build("androidpublisher", "v3", credentials=service_account_credentials)
    result = android_publisher.purchases().products().get(
        packageName=ANDROID_PACKAGE_NAME,
        productId=body.product_id,
        token=body.purchase_token,
    ).execute()

    if result.get("purchaseState") != 0:  # 0 == purchased
        raise HTTPException(400, "Achizitie neconfirmata de Google")

    grant = PRODUCT_GRANTS[body.product_id]
    await apply_grant(user["id"], grant)  # the actual credit/unlock happens ONLY after verification

    await db.processed_purchase_tokens.insert_one({
        "token": body.purchase_token, "user_id": user["id"],
        "product_id": body.product_id, "processed_at": now_iso(),
    })

    # Acknowledge (required within 3 days) and consume if it's a consumable.
    android_publisher.purchases().products().acknowledge(
        packageName=ANDROID_PACKAGE_NAME, productId=body.product_id, token=body.purchase_token,
    ).execute()
    if grant["type"] == "consumable":
        android_publisher.purchases().products().consume(
            packageName=ANDROID_PACKAGE_NAME, productId=body.product_id, token=body.purchase_token,
        ).execute()

    return {"ok": True}
"""
