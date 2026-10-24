TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MODEL_ID = "convnext-tiny-mil-epoch8"
CHECKPOINT_SHA256 = "e30535be63503f489d3e0961afa51cce83bfe7d9766faffbe5dfc2be454310dd"
PIXEL_KEY = "2d0240242985abb81c302fdc9f134933769e3d1032b62f35eed3be9b6f390c79"
NOTICE = "Research demonstration only. Not for diagnosis or patient care. Scores are uncalibrated model outputs."
SLOT_NAMES = ["Sagittal · fat suppressed", "Coronal · fat suppressed", "Axial · fat suppressed",
              "Sagittal · non-fat suppressed", "Coronal · non-fat suppressed"]
TERMINAL = {"completed", "failed", "cancelled", "expired"}
