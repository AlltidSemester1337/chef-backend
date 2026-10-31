// Sign-in for the MCP authorization server (see login.py).
// The password goes to Firebase Auth only; the backend gets an ID token.
// Keep the SDK version in step with FIREBASE_SDK in login.py (the CSP allows gstatic).
import { initializeApp } from "https://www.gstatic.com/firebasejs/13.1.0/firebase-app.js";
import {
  initializeAuth,
  inMemoryPersistence,
  signInWithEmailAndPassword,
} from "https://www.gstatic.com/firebasejs/13.1.0/firebase-auth.js";

const form = document.getElementById("login");
const errorBox = document.getElementById("error");
const { request, apiKey, appId, projectId, authDomain } = form.dataset;

const MESSAGES = {
  "auth/invalid-credential": "Wrong email or password.",
  "auth/invalid-email": "That is not a valid email address.",
  "auth/too-many-requests": "Too many attempts. Wait a while and try again.",
  "auth/network-request-failed": "No connection. Check your network and try again.",
  email_not_verified: "Verify your email address in the Chef app first, then try again.",
  anonymous_not_allowed: "Sign in with an email account.",
  login_expired: "This sign-in link has expired. Go back to the app and connect Chef again.",
};

function showError(code) {
  errorBox.textContent = MESSAGES[code] || "Something went wrong. Try again.";
  errorBox.hidden = false;
}

function setBusy(busy) {
  for (const element of form.elements) element.disabled = busy;
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw { code: payload.detail };
  return payload.redirectUrl;
}

const app = initializeApp({ apiKey, appId, projectId, authDomain });
// Nothing is kept in this origin's storage: the session ends with the page.
const auth = initializeAuth(app, { persistence: inMemoryPersistence });

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  setBusy(true);
  try {
    const credential = await signInWithEmailAndPassword(
      auth,
      form.elements.email.value,
      form.elements.password.value,
    );
    const idToken = await credential.user.getIdToken();
    window.location.assign(await post("/oauth/callback", { request, idToken }));
  } catch (error) {
    showError(error && error.code);
    setBusy(false);
  }
});

document.getElementById("deny").addEventListener("click", async () => {
  setBusy(true);
  try {
    window.location.assign(await post("/oauth/deny", { request }));
  } catch (error) {
    showError(error && error.code);
    setBusy(false);
  }
});
