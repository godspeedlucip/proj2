"""Run the verified projection-to-phase mapping from the ``moni`` folder."""

from utils.phase_mapping import (
    CycleMarker,
    ProjectionPhase,
    build_phase_order,
    build_projection_phase_mapping,
    load_cycle_markers,
    main,
    write_projection_phase_mapping,
)

__all__ = [
    "CycleMarker",
    "ProjectionPhase",
    "build_phase_order",
    "build_projection_phase_mapping",
    "load_cycle_markers",
    "write_projection_phase_mapping",
]


if __name__ == "__main__":
    main()
