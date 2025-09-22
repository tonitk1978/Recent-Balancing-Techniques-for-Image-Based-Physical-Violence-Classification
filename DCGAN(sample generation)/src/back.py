r, c = 3, 5
fig, axs = plt.subplots(r, c, figsize=(10, 6))
fig.suptitle("Generated images", fontsize=20)

noise = np.random.normal(size=(r * c, Z_DIM))
gen_imgs = generator.predict(noise)

cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgs[cnt], cmap="gray_r")
        axs[i, j].axis("off")
        cnt += 1

plt.show()
plt.close()




gen_imgs=gen_imgs* 127.5 + 127.564
#gen_imgs=np.array(object=gen_imgs,dtype=np.uint32)
#gen_imgs=gen_imgs.max()/gen_imgs
#print(gen_imgs[0])
gen_imgs=gen_imgs/gen_imgs.max()
gen_imgs=gen_imgs*255
gen_imgs=np.array(object=gen_imgs,dtype=np.uint32)
print(gen_imgs[0])
print(gen_imgs.max())


print(gen_imgs.shape)
fig, axs = plt.subplots(r, c, figsize=(10, 6))
fig.suptitle("Generated images", fontsize=20)

cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgs[cnt])
        axs[i, j].axis("off")
        cnt += 1

plt.show()
plt.close()




# Suponiendo que gen_imgs es una lista de imágenes (arrays de NumPy)
gen_imgsRes = []  # Usamos una lista para almacenar las imágenes redimensionadas

for img in gen_imgs:
    img = img.astype(np.uint8)  # Asegúrate de que la imagen sea de tipo uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convertir de RGB a BGR
    img = cv2.resize(img, (224, 224))  # Redimensionar la imagen a 224x224
    gen_imgsRes.append(img)  # Agregar la imagen redimensionada a la lista

# Convertir la lista a un array de NumPy si es necesario
gen_imgsRes = np.array(gen_imgsRes, dtype=np.uint8)













fig, axs = plt.subplots(r, c, figsize=(10, 6))
fig.suptitle("Generated images rescaled", fontsize=20)

cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgsRes[cnt])
        axs[i, j].axis("off")
        cnt += 1

plt.show()
plt.close()




print(gen_imgsRes.shape)
