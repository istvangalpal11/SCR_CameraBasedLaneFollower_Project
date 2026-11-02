# Preprocessing constants copied from BFMC tuning.py; values unchanged.

BEV_SRC_TOP_WIDTH_FACTOR            = 0.5    # Source trapezoid top width as image fraction
BEV_SRC_BOTTOM_WIDTH_FACTOR         = 0.95   # Source trapezoid bottom width as image fraction
BEV_SRC_BOTTOM_OFFSET_FACTOR        = 0.0    # Source bottom inset as image-height fraction
BEV_SRC_HORIZON_FACTOR              = 0.71   # Source top position as image-height fraction
BEV_DST_EMPTY_MARGINS               = 0.00   # Destination side margin as output-width fraction

BEV_OUTPUT_WIDTH = 400   # Production BEV destination width in pixels
BEV_OUTPUT_HEIGHT = 400  # Production BEV destination height in pixels

OTSU_VERTICAL_SPLIT_FRAC            = 0.5    # Upper-region height fraction
OTSU_TOP_OFFSET                     = 0      # Manual offset added to upper Otsu threshold
OTSU_BOTTOM_OFFSET                  = 0      # Manual offset added to lower Otsu threshold
OTSU_TOP_AUTO_ADJUST_ENABLE         = True   # Enable automatic upper-threshold increases
OTSU_TOP_MAX_TO_BOTTOM_RATIO        = 1.5    # Maximum upper-to-lower foreground density ratio
OTSU_MIN_BOTTOM_FOREGROUND_RATIO    = 0.005  # Minimum lower density needed as reference
OTSU_TOP_ADJUST_PERCENT             = 0.05   # Fractional increase per automatic upper-threshold step
OTSU_TOP_ADJUST_MAX_STEPS           = 10      # Maximum automatic upper-threshold adjustment steps

THRESH_GAUSSIAN_BLUR_K             = 5      # Fallback Gaussian blur kernel size
THRESH_TOPHAT_ENABLE               = True   # Enable Top-Hat illumination preprocessing
THRESH_TOPHAT_BLUR_K               = 3      # Gaussian blur size before Top-Hat; zero uses fallback
THRESH_TOPHAT_KERNEL_W             = 23     # Top-Hat kernel width in pixels
THRESH_TOPHAT_KERNEL_H             = 1      # Top-Hat kernel height in pixels

THRESH_OPEN_ENABLE                 = False  # Enable morphological opening
THRESH_OPEN_KERNEL_W               = 3      # Opening kernel width in pixels
THRESH_OPEN_KERNEL_H               = 3      # Opening kernel height in pixels
THRESH_OPEN_ITER                   = 1      # Number of opening iterations

THRESH_CLOSE_KERNEL_W              = 7      # Closing kernel width in pixels
THRESH_CLOSE_KERNEL_H              = 35     # Closing kernel height in pixels
THRESH_CLOSE_ITER                  = 1      # Number of closing iterations

BEV_BLACK_TRAPEZOID_ENABLE         = False  # Enable black vehicle-footprint mask in BEV
BEV_BLACK_TRAPEZOID_CENTER_X_FRAC  = 0.50   # Mask center as image-width fraction
BEV_BLACK_TRAPEZOID_TOP_Y_FRAC     = 0.65   # Mask top position as image-height fraction
BEV_BLACK_TRAPEZOID_BOTTOM_Y_FRAC  = 1.00   # Mask bottom position as image-height fraction
BEV_BLACK_TRAPEZOID_TOP_WIDTH_FRAC = 0.4    # Mask top width as image-width fraction
BEV_BLACK_TRAPEZOID_BOTTOM_WIDTH_FRAC = 0.8 # Mask bottom width as image-width fraction