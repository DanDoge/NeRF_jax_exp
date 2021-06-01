### h256

| trail name | baseline PSNR  SSIM | b4c3f9 PSNR  SSIM |
|:-----------|--------------------:|------------------:|
| lego       |    33.07 /     |   34.13 / 0.972   | 
| chair      |    33.86 /     |   34.64 / 0.978   | 
| drums      |    24.95 /     |   25.62 / 0.935   |
| ficus      |    30.38 /     |   31.90 /    | <- config aligned from here
| hotdog     |    36.70 /     |   37.81 /    |
| materials  |    30.00 /     |   31.70 /    |
| mic        |     /     |    /    |
| ship       |     /     |    /    |

lego chair drums ficus hotdog materials mic ship

### h128

| trail name | baseline PSNR  SSIM | b4c3f9 PSNR  SSIM |
|:-----------|--------------------:|------------------:|
| lego       |    30.88 /     |   31.60 / 0.950   | 
| chair      |    31.95 /     |   32.46 / 0.961   | 
| drums      |    24.60 /     |   24.98 / 0.922   |
| ficus      |    29.69 /     |   30.38 / 0.962   |
| hotdog     |    35.51 /     |   36.29 / 0.974   | <-config aligned from here
| materials  |    29.28 /     |   30.36 / 0.957   |
| mic        |    32.67 /     |   32.96 /    |
| ship       |    27.98 /     |   28.39 /    |

### h64, nerf8

| trail name |    gamma x    | gamma x outer |       x       |
|:-----------|--------------:|--------------:|--------------:|
| lego       | 27.96 / 0.903 |       /       | 28.01 / 0.904 | <- not accurate for x
| chair      | 27.64 / 0.915 | 28.06 / 0.909 | 
| drums      | 22.22 / 0.870 | 23.38 / 0.897 | 23.40 / 0.896 | 
| ficus      |       /       |       /       | 27.55 / 0.938 | 
| hotdog     |       /       |       /       | 33.65 / 0.960 | 
| materials  |       /       |       /       | 27.61 / 0.927 | 
| mic        |       /       |       /       |
| ship       |       /       |       /       |
