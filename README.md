# A-Structure-Preserving-Semantic-Transmission-Method-for-Low-Bandwidth-Networks
Semantic-structure-preserving image transmission framework for low-bandwidth scenarios, featuring multi-stream decoupling, structural reconstruction, and efficient low-bitrate communication.
Edge-Link: Structure-Preserving Semantic Transmission for Low-Bandwidth Networks



📌 Introduction



This project provides the implementation of a structure-preserving semantic image transmission method designed for low-bandwidth environments. The method addresses the limitations of traditional compression schemes under ultra-low bitrate conditions and improves structural fidelity, visual quality, and transmission efficiency.Detailed bitrate allocation and object segmentation illustrations can be found in the "paper" folder.



🚀 Key Features



\* Semantic–Structure–Appearance multi-stream decoupling

\* Object-aware semantic rate-distortion optimization

\* Structure-preserving reconstruction

\* Low-latency generative decoding (SPADE-based)

\* Designed for low bandwidth scenarios (e.g., LoRa)



&#x20;📂 Project Structure



Edge-Link-Code/

├── src/                # Source code

│    ├── model/          # Model

│    ├── dataset/        # Dataset files

│    ├── train.py       # Training (fine-tuning) code

│    ├── DIV\_test.py # Inference code for DIV dataset

│    ├──Set14\_test.py# Inference code for Set14 dataset

│    ├── compare\_DIV2K.py# Reconstruction comparison code for DIV2K images

│    ├──compare\_Set14.py# Reconstruction comparison code for Set14 images

├── paper/                # Model Performance Results

├── results/            # Reconstruction results

├── README.md

├── LICENSE

├── requirements.txt

\## ⚙️ Requirements



\* Python 3.8+

\* PyTorch

\* NumPy

\* OpenCV



Install dependencies:



```bash

pip install -r requirements.txt

```



&#x20;▶️ Usage



Run the test script:



```bash

python test.py #example

```

Detailed parameter settings and data paths can be found in the code files.

&#x20;🖼️ Results



Example reconstruction results are provided in the `results/` folder.

&#x20;The results show that, compared with traditional coding methods (e.g., JPEG and WebP), the proposed method exhibits characteristics in terms of structural consistency and visual quality.



&#x20;📊 Dataset



Experiments are conducted on:



\* DIV2K Dataset

\* Set14 Dataset



&#x20;📖 Description



The proposed method decouples image representation into semantic, structural, and appearance streams and performs adaptive bitrate allocation. It ensures high structural consistency and visual naturalness under low-bandwidth constraints.



📌 Notes



This project is part of ongoing research work and is intended for academic and research purposes.





This project is licensed under the MIT License.
