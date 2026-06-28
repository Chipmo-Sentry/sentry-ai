# Alert categories — definitions

Used by both the VLM prompt and `sentry_backend.services.alert_service.derive_alert_level`.

## browsing
**Definition.** Customer is examining merchandise — picking items up, reading labels, holding them, walking between aisles, talking to others. The item stays visible, is returned, or the customer simply moves on.

**Alert level:** `ignore` (never save an evidence clip from VLM alone).

## cart_pickup
**Definition.** Customer takes an item and places it **visibly** into a cart, basket, child stroller, or onto a flat surface that's obviously a "to be purchased" zone. The item remains visible to staff.

**Alert level:** `ignore` (visible basket/cart placement is not theft or an attempt).

## pocket_conceal
**Definition.** Customer takes an item and places it into clothing (pocket, inside jacket), under clothing, under another item, or starts/attempts that concealment motion.

**Alert level:** `ignore` below 0.50, `log` < 0.70, `notify` < 0.85, `review` >= 0.85.

## bag_conceal
**Definition.** Customer takes an item and places it into a personal bag, tote, backpack, pouch, or starts/attempts that concealment motion.

**Alert level:** same as `pocket_conceal`.

## other
**Definition.** Anything not the above categories — staff working, equipment maintenance, a person merely passing through, visible shopping activity, or behavior the VLM can't classify with confidence.

**Alert level:** always `ignore`.
