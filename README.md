♻️ Waste Classification Using Transfer Learning & Fine-Tuning

Classifying waste images as Organic (O) or Recyclable (R) with a pre-trained VGG16 network, and comparing a frozen feature-extraction model against a fine-tuned one.

📚 Course context: Final project of Deep Learning with Keras and TensorFlow (IBM, Coursera). The scenario, dataset, and notebook template were provided by the course; the implementation, training, and evaluation in this repository are my own work on top of that template.

Sample predictions

🎯 Scenario

In the course scenario, a fictional waste management organization (EcoClean) wants to automate waste sorting, since manual sorting is slow, error-prone, and leads to contamination of recyclable materials. The goal is an image classifier that separates recyclable from organic waste.

📦 Dataset

Provided by the course (downloaded automatically by the notebook). It appears to be a reduced version of the public Waste Classification Data dataset on Kaggle.

Split	Images	Notes
Train	800	80% of the training folder (validation_split = 0.2)
Validation	200	20% of the training folder
Test	200	separate folder; evaluated on a balanced 100-image subset (first 50 O + first 50 R)

Classes: O (organic) and R (recyclable).

🧠 Approach
Preprocessing & augmentation with Keras ImageDataGenerator: images resized to 150×150 and rescaled to [0, 1]; training augmentation with width/height shifts (0.1) and horizontal flips.
Feature extraction: VGG16 (ImageNet weights, include_top=False) fully frozen, followed by a custom head: Flatten → Dense(512) → Dropout(0.3) → Dense(512) → Dropout(0.3) → Dense(1, sigmoid).
Fine-tuning: same architecture, but the last convolutional layer (block5_conv3) is unfrozen and trained together with the head.
Evaluation: accuracy/loss curves for both models, precision/recall/F1 on test images, and visual inspection of individual predictions.
Training setup
Setting	Value
Image size	150 × 150
Batch size	32
Epochs	10 (5 steps per epoch)
Loss	Binary cross-entropy
Optimizer	RMSprop, initial LR 1e-4 with exponential decay (lr = 1e-4 · e^(−0.1·epoch))
Callbacks	Early stopping (patience 4 on validation loss) and best-model checkpointing
Random seed	42
📊 Results
Model	Train Acc (epoch 10)	Val Acc (epoch 10)	Val Loss (epoch 10)	Test Acc
VGG16 – Feature Extraction	83.1%	89.1%	0.344	86%
VGG16 – Fine-Tuned	90.0%	89.6%	0.251	85%

Test metrics (100 images, 50 per class):

Model	Class	Precision	Recall	F1-score
Feature Extraction	O	0.85	0.88	0.86
Feature Extraction	R	0.88	0.84	0.86
Fine-Tuned	O	0.84	0.86	0.85
Fine-Tuned	R	0.86	0.84	0.85

Observations

Fine-tuning improved training accuracy and lowered validation loss (0.344 → 0.251), but did not improve test accuracy (85% vs 86%). With only 100 test images, a 1-point gap equals a single image, so the two models perform essentially the same here.
Both models perform evenly across the two classes, with no strong bias toward either.
Each epoch used only 5 batches (160 of the 800 training images), so the models were probably under-trained, which may explain why fine-tuning showed little benefit on the test set.
🛠️ Tech Stack

Python · TensorFlow 2.17 / Keras · NumPy · Matplotlib · scikit-learn · tqdm · Jupyter Notebook

🚀 How to Run

The notebook downloads the dataset automatically from the course-hosted link, so no manual data setup is needed. After downloading or cloning this repository, open a terminal in its folder and run:

bash
pip install tensorflow==2.17.0 numpy==1.26.0 scikit-learn==1.5.1 matplotlib==3.9.2 requests tqdm
jupyter notebook Final_Proj_Classify_Waste_Products_Using_TL_FT.ipynb

Training VGG16 without a GPU is slow (about 80 seconds per epoch in my run). A GPU runtime such as Google Colab is recommended.

📁 Repository Structure
├── Final_Proj_Classify_Waste_Products_Using_TL_FT.ipynb   # Main notebook
├── final_proj_classify_waste_products_using_tl_ft.py      # Script version
├── Screenshot .png                                         # Sample output
└── README.md
🔮 Future Improvements
Train on the full training set in every epoch (more steps per epoch / more epochs) and unfreeze more VGG16 layers
Evaluate on the full 200-image test set and add a confusion matrix
Try other backbones (ResNet50, EfficientNet, MobileNetV2)
Extend to multi-class waste categories (plastic, glass, paper, metal)
Deploy as a small web app (Streamlit / Gradio)
🙏 Acknowledgments

Project structure, scenario, and dataset provided by the Deep Learning with Keras and TensorFlow course (IBM, Coursera). The dataset appears to be a reduced version of the public Waste Classification Data on Kaggle.

👩‍💻 Author

Ghaydaa-Aldirawi
