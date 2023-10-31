
class Eval_image:
  def __init__(self, model_interf, img_path, output_dir,root_img_path):
    if isinstance(model_interf, str) and model_interf.endswith("h5"):
      model = tf.keras.models.load_model(model_interf)
      self.model_interf = lambda imms: model((imms - 127.5) * 0.0078125).numpy()
    else:
      self.model_interf = model_interf
    self.dist_func = lambda aa, bb: np.dot(aa, bb) # hàm tính khoảng cách giữa 2 vector
    self.output_dir = output_dir
    self.img_path = img_path
    print('begin serving')
    self.root_path = root_img_path
    self.embs, self.imm_classes, self.filenames = self.prepare_image_and_embedding(self.img_path, self.output_dir)

  def prepare_image_and_embedding(self, img_folder, output_dir):
    save_embeddings = output_dir
    # kiểm tra đường dẫn, nếu đã lưu các embedding vector vào file npz thì chỉ load lại
    if save_embeddings and os.path.exists(save_embeddings):
      print(">>>> Reloading from backup:", save_embeddings)
      aa = np.load(save_embeddings)
      embs, imm_classes, names = aa["embs"], aa["imm_classes"], aa["filenames"]
      embs, img_classes = embs.astype("float32"), imm_classes.astype("int")
    else:
      # chưa lưu => đọc mỗi hình
      img_shape = (112, 112) #GhostFaceNet yêu cầu ảnh input có kích thước 112*112
      imgs = pd.read_csv('./data.csv')['image'].tolist() # danh sách tên các hình
      labels = pd.read_csv('./data.csv')['label'].tolist() # danh sách label của các hình
      embs = [] # list of embeddings
      img_classes = [] # list nhãn của mỗi hình (mỗi vector embedding)
      names = [] # list lưu tên file của mỗi hình
      for i,img_path in enumerate(tqdm(imgs)):
        img_path = os.path.join(self.root_path, img_path) # đọc file từng ảnh
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_shape, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) # biến ảnh về kích thước (1,112,112,3)

        emb = self.model_interf(img) # lưu vector embedding
        emb = normalize(np.array(emb).astype("float32"))[0] # chuẩn hoá 1 vector theo L2 norm
        img_class = int(labels[i]) # nhãn của ảnh hiện tại
        img_classes.append(img_class) # lưu nhãn của ảnh hiện tại
        embs.append(emb) # lưu vector embedding của ảnh hiện tại
        filename = os.path.basename(img_path) # lấy filename (dạng abc.xyz)
        names.append(filename) # lưu filename của ảnh hiện tại
      # lưu tất cả thông tin trên vào 1 file npy
      np.savez(output_dir, embs=embs, imm_classes=img_classes, filenames=names)
      print("file chứa danh sách vector được save tại")
    return np.array(embs),np.array(img_classes),np.array(names)


  def do_evaluation(self):
    ## đếm xem có bao nhiêu identity trong dataset aka có bao nhiêu người có mặt trong dataset :D
    register_ids = np.unique(self.imm_classes)
    print(">>>> [base info] embs:", self.embs.shape, "imm_classes:", self.imm_classes.shape, "register_ids:", register_ids.shape)

    register_base_embs = np.array([]).reshape(0, self.embs.shape[-1]) # lưu vector đại diện của từng identity
    register_base_dists = []
    for register_id in tqdm(register_ids, "Evaluating"):
      # với từng identity: tìm các hình chứa gương mặt của người đó
      pos_pick_cond = self.imm_classes == register_id
      # lấy vector của các gương mặt đó
      pos_embs = self.embs[pos_pick_cond]
      # tạo ra vector đại diện cho identity đó : sum các vector gương mặt, sau đó normalize
      register_base_emb = normalize([np.sum(pos_embs, 0)])[0]
      # tính similarity của vector đại diện tới toàn bộ vector trong dataset
      register_base_dist = self.dist_func(self.embs, register_base_emb)
      register_base_dists.append(register_base_dist) # lưu kết quả vào 1 list
      register_base_embs = np.vstack([register_base_embs, register_base_emb])

    ## save 1000 vector đại diện
    #new_path = self.output_dir.replace(os.path.basename(self.output_dir),"processed_embedding.npz")
    #print("\n Saving vector đại diện tại: ", new_path)
    #np.savez(new_path,embs =register_base_embs)

    register_base_dists = np.array(register_base_dists).T # similarity (cosine similarity) của toàn bộ identity tới toàn bộ vector trong dataset
    # cosine similarity from one image to each class
    print(register_base_dists.shape) # như vậy nó có shape 4720x1000

    # đồng thời theo chiều ngược lại : 1 ảnh (trong 4720 ảnh trong dataset), sẽ dc tính độ tương tự với 1000 vector đại diện
    ### bắt đầu quá trình tính toán accuracy

    # register_base_dists.argmax(1) => tìm identity tương tự nhất cho mỗi ảnh
    # accuracy = số lượng ảnh mà model detect identity đúng / số lượng ảnh (4720)
    accuracy = (register_base_dists.argmax(1) == self.imm_classes).sum() / register_base_dists.shape[0]
    print("register_ids shape: ",register_ids.shape)
    print("self.imm_classes shape: ",self.imm_classes.shape)

    # tạo 1 array có 4720 dòng x 1000 cột => mỗi dòng là 1 hình, hình dc predict có identity nào thì giá trị cột đó =1
    reg_pos_cond = np.equal(register_ids, np.expand_dims(self.imm_classes, 1))
    print(reg_pos_cond.shape) # array co 1 nghin cot, gia tri tai cot label =1

    # tính similarity của hình đó với vector identity gần nhất (positive prediction)
    reg_pos_dists = register_base_dists[reg_pos_cond].ravel()
    print(reg_pos_dists.shape) # 4720
    # similarity của các hình với các 999 vector identity xa hơn (999 negative prediction)
    reg_neg_dists = register_base_dists[np.logical_not(reg_pos_cond)].ravel()
    print(reg_neg_dists.shape) # 4720*999
    # turn it into a binary classification
    # label: tất cả prediction của model (positive hay negative) đều dc flatten
    # score: similarity tương ứng của model cho các quyết định đó
    label = np.concatenate([np.ones_like(reg_pos_dists), np.zeros_like(reg_neg_dists)])
    score = np.concatenate([reg_pos_dists, reg_neg_dists]) #distance tai cot co label va k co label
    return accuracy,score,label