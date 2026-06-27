img = [0] * 784

for row in range(28):
    img[row * 28 + 14] = 255
    img[row * 28 + 13] = 200  
    img[row * 28 + 15] = 200

with open('digit1.bin', 'wb') as f:
    f.write(bytes(img))
