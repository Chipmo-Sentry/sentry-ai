# Alert categories — definitions

Used by both the VLM prompt and `sentry_backend.services.alert_service.derive_alert_level`.

## browsing
**Definition.** Customer is examining merchandise — picking items up, reading labels, holding them, walking between aisles, talking to others. **Eventually puts the item back or moves on without claiming it.** Most common store activity.

**Alert level:** `ignore` (the AI level mapping ignores browsing regardless of confidence).

## cart_pickup
**Definition.** Customer takes an item and places it **visibly** into a cart, basket, child stroller, or onto a flat surface that's obviously a "to be purchased" zone. The item remains visible to staff.

**Alert level:** `log` if confidence < 0.70, else `notify`.

## pocket_conceal
**Definition.** Customer takes an item and places it into clothing (pocket, inside jacket), a personal bag, under another item, or otherwise hides it from camera view. Often paired with **looking around** behavior immediately before or after.

**Alert level:** `log` < 0.70 < `notify` < 0.85 ≤ `review`.

## other
**Definition.** Anything not the above three — staff working, equipment maintenance, a person merely passing through, or behavior the VLM can't classify with confidence.

**Alert level:** always `log` (recorded, not pushed).
