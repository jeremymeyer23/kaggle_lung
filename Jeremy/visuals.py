# Example EDA
plt.style.use('seaborn')

# Batches of 20
row, col, batch = (5, 4, 1)
IDs = train['Patient'].unique()[row*col*(batch-1):row*col*batch]
fig, axs = plt.subplots(row, col, figsize=(12, 15))
for i in range(row):
    for j in range(col):
        if(col*i+j >= len(IDs)): break
        id = IDs[col*i+j]
        train_id = train[train['Patient'] == id]
        axs[i, j].plot(train_id['Weeks'], train_id['FVC'], 'ro-')
        axs[i, j].set_title('Patient ID: {}'.format(id), fontdict={'fontsize': mpl.rcParams['axes.titlesize']*.7})

# Hide x labels and tick labels for top and right plots
for ax in axs.flat:
    ax.set(xlabel='Week', ylabel='FVC Score')
    ax.label_outer()
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)
plt.tight_layout(h_pad=4.5, w_pad=1.5)
plt.show()

# Pixel plot
def plt_hist(id=0):
    path = 'data/train/' + patients[id]
    first_patient = load_scan(path)
    first_patient_pixels = get_pixels_hu(first_patient)
    plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()


# Show some slice in the middle
def plt_cross_section(hu_slices, i):
    plt.style.use('default')
    plt.imshow(hu_slices[i], cmap=plt.cm.gray)
    plt.show()

# 3D plot
def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


plot_3d(pix_resampled, 400)