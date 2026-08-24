"""
Authentication: email+password registration/login, server-side sessions,
and abuse-prevention rate limits.

SESSION MODEL (not a stateless JWT):
A JWT alone can't be revoked or checked against "last used" without a
database lookup anyway, so we make that lookup the source of truth and use
the JWT only as a signed, unguessable session ID. Every authenticated
request:
  1. decodes the JWT to get the session id (fast, cheap, catches tampering)
  2. loads the session doc from Mongo
  3. checks: not revoked, not past absolute expiry, not past inactivity
     expiry (now - last_used_at < 30 days)
  4. updates last_used_at and appends the current IP to the session's IP
     history (for anomaly detection — e.g. Russia today, India tomorrow)
  5. rejects if the current IP is on the ban list

Expiry rules (as specified):
- Inactivity expiry: 30 days since the token was last used. Using it
  resets the clock, so an active user is never logged out.
- Absolute expiry: 7 years since login, regardless of activity. A safety
  ceiling so no session lives forever even if used daily forever.

Login rate limits (as specified):
- Max 50 login attempts per IP per rolling 50-hour window (any more than
  that from one IP is not realistic for real humans — bot territory).
- Max 5 login attempts per device_id per rolling 24-hour window.

Signup limit (as specified):
- Max 10 accounts created per IP (lifetime cap).
"""

import os
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
import jwt
from fastapi import HTTPException, Header, Request
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET nu este setat in .env — genereaza unul (ex: `openssl rand -hex 32`) "
        "si adauga-l ca JWT_SECRET=... inainte de a porni serverul."
    )
JWT_ALGORITHM = "HS256"

SESSION_INACTIVITY_DAYS = 30
SESSION_ABSOLUTE_MAX_DAYS = 7 * 365

LOGIN_IP_LIMIT = 50
LOGIN_IP_WINDOW_HOURS = 50
LOGIN_DEVICE_LIMIT = 5
LOGIN_DEVICE_WINDOW_HOURS = 24
MAX_ACCOUNTS_PER_IP = 10

MAX_IP_HISTORY_PER_SESSION = 50


def now():
    return datetime.now(timezone.utc)


def now_iso():
    return now().isoformat()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    device_id: Optional[str] = None


class LoginIn(BaseModel):
    email: EmailStr
    password: str
    device_id: Optional[str] = None


class AuthModule:
    def __init__(self, db):
        self.db = db
        self.users = db.users
        self.sessions = db.sessions
        self.login_attempts = db.login_attempts
        self.signup_ips = db.signup_ips
        self.banned_ips = db.banned_ips

    async def ensure_indexes(self):
        await self.users.create_index("email", unique=True)
        await self.sessions.create_index("id", unique=True)
        await self.sessions.create_index("user_id")
        await self.login_attempts.create_index(
            "ts", expireAfterSeconds=LOGIN_IP_WINDOW_HOURS * 3600 + 3600
        )
        await self.signup_ips.create_index("ip")
        await self.banned_ips.create_index("ip", unique=True)

    async def is_ip_banned(self, ip: str) -> bool:
        doc = await self.banned_ips.find_one({"ip": ip})
        return doc is not None

    async def ban_ip(self, ip: str, reason: str = ""):
        await self.banned_ips.update_one(
            {"ip": ip},
            {"$set": {"ip": ip, "reason": reason, "banned_at": now_iso()}},
            upsert=True,
        )

    async def unban_ip(self, ip: str):
        await self.banned_ips.delete_one({"ip": ip})

    async def _check_login_rate_limits(self, ip: str, device_id: Optional[str]):
        ip_window_start = now() - timedelta(hours=LOGIN_IP_WINDOW_HOURS)
        ip_count = await self.login_attempts.count_documents({
            "ip": ip, "ts": {"$gte": ip_window_start.isoformat()},
        })
        if ip_count >= LOGIN_IP_LIMIT:
            raise HTTPException(
                429,
                "Prea multe incercari de autentificare de la aceasta adresa IP. "
                "Incearca din nou peste cateva ore."
            )

        if device_id:
            device_window_start = now() - timedelta(hours=LOGIN_DEVICE_WINDOW_HOURS)
            device_count = await self.login_attempts.count_documents({
                "device_id": device_id, "ts": {"$gte": device_window_start.isoformat()},
            })
            if device_count >= LOGIN_DEVICE_LIMIT:
                raise HTTPException(
                    429,
                    "Prea multe incercari de autentificare de pe acest dispozitiv. "
                    "Incearca din nou peste 24 de ore, sau reseteaza-ti parola."
                )

    async def _record_login_attempt(self, ip: str, device_id: Optional[str], success: bool):
        await self.login_attempts.insert_one({
            "id": str(uuid.uuid4()), "ip": ip, "device_id": device_id,
            "success": success, "ts": now_iso(),
        })

    async def _check_signup_ip_limit(self, ip: str):
        count = await self.signup_ips.count_documents({"ip": ip})
        if count >= MAX_ACCOUNTS_PER_IP:
            raise HTTPException(
                429,
                f"S-a atins limita de {MAX_ACCOUNTS_PER_IP} conturi create de la aceasta adresa IP."
            )

    async def _create_session(self, user_id: str, ip: str) -> str:
        session_id = str(uuid.uuid4())
        session_doc = {
            "id": session_id,
            "user_id": user_id,
            "created_at": now_iso(),
            "last_used_at": now_iso(),
            "ip_history": [{"ip": ip, "ts": now_iso()}],
            "revoked": False,
        }
        await self.sessions.insert_one(session_doc)
        token = jwt.encode({"sid": session_id}, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return token

    async def _touch_session(self, session_doc: dict, ip: str):
        update = {"$set": {"last_used_at": now_iso()}}
        last_ip = session_doc["ip_history"][-1]["ip"] if session_doc.get("ip_history") else None
        if ip != last_ip:
            update["$push"] = {
                "ip_history": {
                    "$each": [{"ip": ip, "ts": now_iso()}],
                    "$slice": -MAX_IP_HISTORY_PER_SESSION,
                }
            }
        await self.sessions.update_one({"id": session_doc["id"]}, update)

    async def register(self, body: RegisterIn, ip: str) -> dict:
        if await self.is_ip_banned(ip):
            raise HTTPException(403, "Aceasta adresa IP este blocata.")

        await self._check_signup_ip_limit(ip)

        existing = await self.users.find_one({"email": body.email.lower()})
        if existing:
            raise HTTPException(409, "Exista deja un cont cu acest email.")

        if len(body.password) < 8:
            raise HTTPException(400, "Parola trebuie sa aiba cel putin 8 caractere.")

        user_id = str(uuid.uuid4())
        user_doc = {
            "id": user_id,
            "email": body.email.lower(),
            "password_hash": hash_password(body.password),
            "created_at": now_iso(),
        }
        await self.users.insert_one(user_doc)
        await self.signup_ips.insert_one({
            "id": str(uuid.uuid4()), "ip": ip, "user_id": user_id, "ts": now_iso(),
        })

        token = await self._create_session(user_id, ip)
        return {"token": token, "user": {"id": user_id, "email": body.email.lower()}}

    async def login(self, body: LoginIn, ip: str) -> dict:
        if await self.is_ip_banned(ip):
            raise HTTPException(403, "Aceasta adresa IP este blocata.")

        await self._check_login_rate_limits(ip, body.device_id)

        user = await self.users.find_one({"email": body.email.lower()})
        if not user or not verify_password(body.password, user["password_hash"]):
            await self._record_login_attempt(ip, body.device_id, success=False)
            raise HTTPException(401, "Email sau parola incorecte.")

        await self._record_login_attempt(ip, body.device_id, success=True)
        token = await self._create_session(user["id"], ip)
        return {"token": token, "user": {"id": user["id"], "email": user["email"]}}

    async def get_current_user(self, authorization: Optional[str], ip: str) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Autentificare necesara.")
        token = authorization[len("Bearer "):]

        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.InvalidTokenError:
            raise HTTPException(401, "Token invalid.")

        session_id = payload.get("sid")
        if not session_id:
            raise HTTPException(401, "Token invalid.")

        session_doc = await self.sessions.find_one({"id": session_id})
        if not session_doc or session_doc.get("revoked"):
            raise HTTPException(401, "Sesiune invalida — te rog autentifica-te din nou.")

        if await self.is_ip_banned(ip):
            raise HTTPException(403, "Aceasta adresa IP este blocata.")

        created_at = datetime.fromisoformat(session_doc["created_at"])
        if now() - created_at > timedelta(days=SESSION_ABSOLUTE_MAX_DAYS):
            raise HTTPException(401, "Sesiune expirata — te rog autentifica-te din nou.")

        last_used_at = datetime.fromisoformat(session_doc["last_used_at"])
        if now() - last_used_at > timedelta(days=SESSION_INACTIVITY_DAYS):
            raise HTTPException(401, "Sesiune expirata din inactivitate — te rog autentifica-te din nou.")

        user = await self.users.find_one({"id": session_doc["user_id"]})
        if not user:
            raise HTTPException(401, "Utilizator inexistent.")

        await self._touch_session(session_doc, ip)

        return {"id": user["id"], "email": user["email"]}

    async def get_session_ip_history(self, user_id: str) -> list:
        sessions = await self.sessions.find({"user_id": user_id}).to_list(200)
        history = []
        for s in sessions:
            history.extend(s.get("ip_history", []))
        history.sort(key=lambda x: x["ts"], reverse=True)
        return history
