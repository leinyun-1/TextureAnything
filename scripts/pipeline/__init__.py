"""Pair factory.

catalog -> l1 -> l2 -> acquire -> render_p0 -> sfm_p0 -> render_p4 -> sfm_p4 -> gate -> package

Each stage reads the ledger, advances eligible records, and leaves rejects in place.
Rendering and COLMAP call the existing scripts. Projector renders are not part of this factory.
"""
