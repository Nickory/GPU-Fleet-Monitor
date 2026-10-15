#!/bin/zsh
set -euo pipefail

MAC_APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MAC_APP_DIR/.." && pwd)"
APP_DIR="$PROJECT_DIR/dist/GPU Fleet Monitor.app"
ZIP_PATH="$PROJECT_DIR/dist/GPU Fleet Monitor.zip"
DERIVED_DATA_DIR="$PROJECT_DIR/.build/xcode"

xcodebuild \
  -quiet \
  -project "$MAC_APP_DIR/GPUFleet.xcodeproj" \
  -scheme GPUFleetMonitor \
  -configuration Release \
  -derivedDataPath "$DERIVED_DATA_DIR" \
  CODE_SIGNING_ALLOWED=YES \
  CODE_SIGNING_REQUIRED=YES \
  CODE_SIGN_IDENTITY=- \
  ARCHS=arm64 \
  build

PRODUCT_APP="$DERIVED_DATA_DIR/Build/Products/Release/GPU Fleet Monitor.app"
mkdir -p "$PROJECT_DIR/dist"
rm -rf "$APP_DIR"
ditto "$PRODUCT_APP" "$APP_DIR"
plutil -lint "$APP_DIR/Contents/Info.plist"
codesign --verify --deep --strict "$APP_DIR"
rm -f "$ZIP_PATH"
ditto -c -k --sequesterRsrc --keepParent "$APP_DIR" "$ZIP_PATH"
printf 'Built: %s\n' "$APP_DIR"
printf 'Packed: %s\n' "$ZIP_PATH"
