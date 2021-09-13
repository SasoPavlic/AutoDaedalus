# AutoDaedalus
### Description 📝
The design of neural network architecture is becoming more difficult as the complexity of the problems we tackle using machine learning increases. Many variables influence the performance of a neural model, and those variables are often limited by the researcher's prior knowledge and experience. In our master's thesis, we will focus on becoming familiar with evolutionary neural network design, anomaly detection techniques, and a deeper knowledge of autoencoders and their potential for application in unsupervised learning. Our practical objective will be to build a neural architecture search based on swarm intelligence, and construct an autoencoder architecture for anomaly detection in the MNIST dataset.

### What it can do? 👀
* **Construct novel autoencoder's architecture** using neural architecture search (NAS) based on ant colony optimization (Swarm intelligence)
* Allow an **unsupervised machine learning algorithm** to make decisions that mark the threshold between normal and anomalous data instances. 
* **Finds anomalies** in MNIST dataset based on configuration parameters  

### Requirements ✅
* **CUDA supported GPU**
* **CUDA** release 11.1
* **Tensorflow** 2.4.1
* **Keras** 2.3.1
* **Python** 3.8
##### INSTALL ALL
pip3 install -r requirements.txt

### Documentation 📘 
* Master's thesis (link will be available shortly...)

### Usage 🔨
Configure `settings/autoencoder.yaml` according to your needs.
Run script with Python`autodaedalus_main.py`

### Flowchart 📝

<p align="center">
  <img src="https://user-images.githubusercontent.com/9087174/55276558-066c5300-52ed-11e9-8bb6-284948cdef67.png" width="300">
</p>

<p align="center">
  <strong>Flowchart representing AutoDaedalus workflow</strong>
</p>

## Future goals 🌟
- [ ] Adjust input node to accept any dataset automatic (CIFAR-10, Fashion mnist, ...)
- [ ] Split functions into smaller components
- [ ] Improve logging and final results readability


## Acknowledgments 🎓

AutoDaedalus was developed under the supervision of [doc. dr Sašo Karakatič](https://ii.feri.um.si/en/person/saso-karakatic-2/)  for the Master degree of Informatics and Technologies of Communication in [University of Maribor](https://www.um.si/en/Pages/default.aspx).
