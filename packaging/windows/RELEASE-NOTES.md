# Preview 23 — 2026-09-16

OVR playspace movement no longer gets cancelled by tracker coordinate conversion. A live physical reference is captured at alignment/confirmation; outgoing tracker positions, orientations and velocities use this reference. Incoming VR observations are converted back to the same reference before body correction. Ordinary same-room chaperone changes no longer invalidate alignment; runtime restarts and hardware universe changes still do. The optional OSC path receives equivalent current-standing coordinates.

Existing saved calibration formats are compatible. The live reference is intentionally not saved across sessions: reset OVR offsets before confirming saved alignment after a restart. Alignment and proportions remain user-paced/countdown-driven as in preview 22.

Model selection and SDK baseline comparisons moved to Advanced. The model menu shows only models installed in that folder. The full distribution keeps optimized SAM plus the original SAM alternative; old intermediate FP8/NLF experiments, backup executables, historical notes and developer tools are not included. Existing development and personal files are kept outside the clean distribution.

Validation: all ten automated suites passed. Regression tests cover all three trackers under vertical drag, yaw+translation, reset, stale/restarted references, controller conversion, calibration during drag, and full body/contact tracking at 15/22/30 Hz. The 301-frame cached replay has unchanged tracking data (processing time excluded). Live OVR/VRChat behavior has not been confirmed by this validation.
