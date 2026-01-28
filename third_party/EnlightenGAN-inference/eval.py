from enlighten_inference import EnlightenOnnxModel
import cv2

# by default, CUDAExecutionProvider is used
# model = EnlightenOnnxModel()
# however, one can choose the providers priority, e.g.: 
model = EnlightenOnnxModel(providers = ["CPUExecutionProvider"])

img = cv2.imread('./tests/R-001.png')
processed = model.predict(img)
cv2.imwrite('./R-001-P.png', processed)

img = cv2.imread('./tests/R-002.png')
processed = model.predict(img)
cv2.imwrite('./R-002-P.png', processed)

img = cv2.imread('./tests/R-003.png')
processed = model.predict(img)
cv2.imwrite('./R-003-P.png', processed)

img = cv2.imread('./tests/R-004.png')
processed = model.predict(img)
cv2.imwrite('./R-004-P.png', processed)

img = cv2.imread('./tests/R-005.png')
processed = model.predict(img)
cv2.imwrite('./R-005-P.png', processed)
