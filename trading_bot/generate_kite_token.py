"""
Helper script to exchange Zerodha KiteConnect request_token for access_token.

Usage:
    python generate_kite_token.py --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET --request-token YOUR_REQUEST_TOKEN
"""

import argparse
from kiteconnect import KiteConnect


def main():
    parser = argparse.ArgumentParser(description="Generate Zerodha KiteConnect Access Token")
    parser.add_argument("--api-key", required=True, help="Your KiteConnect API Key")
    parser.add_argument("--api-secret", required=True, help="Your KiteConnect API Secret")
    parser.add_argument("--request-token", required=True, help="Request Token from login redirect URL")

    args = parser.parse_args()

    try:
        kite = KiteConnect(api_key=args.api_key)
        data = kite.generate_session(args.request_token, api_secret=args.api_secret)

        access_token = data["access_token"]
        print("\n" + "=" * 60)
        print("KiteConnect Session Generated Successfully! 🎉")
        print("=" * 60)
        print(f"API Key       : {args.api_key}")
        print(f"Access Token  : {access_token}")
        print("=" * 60)
        print("\nYou can now run live paper/real trading with:")
        print(f"python main.py paper --source kite --api-key {args.api_key} --access-token {access_token}\n")

    except Exception as e:
        print(f"\n[Error] Failed to generate access token: {e}")


if __name__ == "__main__":
    main()
