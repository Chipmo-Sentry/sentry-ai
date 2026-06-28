"""Learned skeleton-action anomaly model (ADR-0030, Фаз 1).

Bootstrap = anomaly-first: shoplifting is rare, so instead of a supervised
theft/benign classifier (which needs many positive examples we don't have yet),
we learn what NORMAL shopping motion looks like from pose sequences and flag
DEVIATION. A compact temporal autoencoder reconstructs normal pose windows well
and badly on unusual ones — reconstruction error IS the anomaly score.

Edge-friendly by construction: input is skeletons only (no pixels — privacy), the
model is a few hundred KB, and it runs on CPU. Trained centrally (PoseLift, then
per-store verified clips); the same roc_auc / build_pose_report from eval.pose
scores it head-to-head against the rule-based BehaviorScorer baseline, so it's
promoted only if it actually beats the engine (eval-gated shadow rollout).
"""
