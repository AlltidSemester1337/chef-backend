"""Grant or revoke the Firebase custom claim {"admin": true} for one user.

Admins have no beta quota. Runs locally with Application Default Credentials:

    gcloud auth application-default login
    gcloud auth application-default set-quota-project your-gcp-project-id
    uv run python scripts/set_admin_claim.py --project your-gcp-project-id --email you@example.com

The claim reaches the user's ID token on the next token refresh (sign out and in,
or wait up to an hour).
"""

import argparse

import firebase_admin  # pyright: ignore[reportMissingTypeStubs]
from firebase_admin import auth  # pyright: ignore[reportMissingTypeStubs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--project", required=True, help="GCP/Firebase project ID")
    parser.add_argument("--email", required=True, help="email address of the user")
    parser.add_argument("--revoke", action="store_true", help="remove the admin claim instead")
    args = parser.parse_args()

    app = firebase_admin.initialize_app(options={"projectId": args.project})
    user = auth.get_user_by_email(args.email, app=app)
    # set_custom_user_claims replaces all claims, so keep any others the user has.
    claims = dict(user.custom_claims or {})
    if args.revoke:
        claims.pop("admin", None)
    else:
        claims["admin"] = True
    auth.set_custom_user_claims(user.uid, claims or None, app=app)
    print(f"uid {user.uid}: custom claims are now {claims or '{}'}")


if __name__ == "__main__":
    main()
