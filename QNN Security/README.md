# 🔒 QNN Security - Quantum Neural Network for Network Intrusion Detection

## 📋 Project Overview

This project implements a **Quantum Neural Network (QNN)** for network intrusion detection using the NF-UNSW-NB15 dataset. The goal is to leverage quantum computing principles to enhance cybersecurity by detecting malicious network traffic patterns with improved accuracy and efficiency.

## 🎯 Objectives

- **Quantum Advantage**: Explore quantum computing's potential in cybersecurity
- **Network Security**: Detect and classify network intrusions using quantum algorithms
- **Performance Optimization**: Compare quantum vs classical approaches for intrusion detection
- **Feature Engineering**: Develop quantum-specific feature encoding methods

## 📊 Dataset

**NF-UNSW-NB15 Dataset**
- **Size**: 17MB compressed, comprehensive network traffic data
- **Features**: Network flow characteristics (protocol, bytes, packets, duration, etc.)
- **Classes**: Binary classification (Benign vs Malicious traffic)
- **Challenge**: Imbalanced dataset requiring careful preprocessing

## 🏗️ Architecture

### Quantum Neural Network Components

1. **Quantum Feature Encoding**
   - Classical features → Quantum state representation
   - Normalization and quantization to quantum amplitudes
   - Feature selection: `PROTOCOL`, `L7_PROTO`, `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`, `TCP_FLAGS`, `FLOW_DURATION_MILLISECONDS`

2. **Quantum Circuit Design**
   - TensorFlow Quantum (TFQ) framework
   - Cirq for quantum circuit construction
   - Parameterized quantum gates for learning

3. **Hybrid Classical-Quantum Training**
   - Classical optimizer with quantum gradients
   - End-to-end differentiable quantum circuits
   - Backpropagation through quantum layers

## 📁 Project Structure

```
QNN Security/
├── v0_dataCheck_QNN.ipynb     # Initial data exploration and setup
├── v1_dataCheck_QNN.ipynb     # Amazon Braket integration attempt
├── v2_QNN.ipynb               # TensorFlow Quantum implementation
├── v3_QNN.ipynb               # Final optimized QNN model
├── NF-UNSW-NB15.zip           # Network security dataset
└── Instruction for Quantum Neural Network Implementation.docx
```

## 🔧 Implementation Versions

### v0: Initial Setup
- Basic TensorFlow Quantum installation
- Dataset loading and exploration
- Initial feature preprocessing

### v1: Amazon Braket Integration
- Attempted AWS quantum computing integration
- Alternative quantum computing approach
- Cloud-based quantum simulation

### v2: TensorFlow Quantum Framework
- Full TFQ implementation
- Quantum circuit design
- Hybrid classical-quantum training

### v3: Optimized QNN Model
- **Final implementation** with best practices
- Advanced feature encoding
- Balanced dataset handling
- Performance optimization

## 🚀 Key Features

### Data Preprocessing
```python
# Feature selection and balancing
selected_features = [
    'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES',
    'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS'
]

# Quantum feature encoding
def encode_features(df, features):
    df_encoded = df.copy()
    for feature in features:
        max_value = df_encoded[feature].max()
        df_encoded[feature] = df_encoded[feature] / max_value * np.pi
        df_encoded[feature] = np.round(df_encoded[feature] / 0.25) * 0.25
    return df_encoded
```

### Quantum Circuit Design
- **Parameterized quantum gates** for learning
- **Quantum feature maps** for data encoding
- **Measurement operations** for classification

### Training Pipeline
1. **Data Balancing**: Resample imbalanced classes
2. **Feature Encoding**: Classical → Quantum representation
3. **Quantum Training**: Hybrid classical-quantum optimization
4. **Evaluation**: Performance metrics and comparison

## 🛠️ Technical Stack

### Core Libraries
- **TensorFlow Quantum**: Quantum machine learning framework
- **Cirq**: Quantum circuit construction
- **TensorFlow**: Classical neural network components
- **Pandas/NumPy**: Data manipulation and preprocessing
- **Scikit-learn**: Classical ML comparison and utilities

### Quantum Computing
- **TFQ**: TensorFlow Quantum for hybrid quantum-classical models
- **Cirq**: Google's quantum circuit library
- **SymPy**: Symbolic computation for quantum operations

## 📈 Performance Metrics

### Evaluation Criteria
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Balanced performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Quantum Advantage**: Comparison with classical approaches

### Expected Outcomes
- Enhanced detection accuracy for network intrusions
- Improved performance on imbalanced datasets
- Quantum computational advantages in specific scenarios

## 🔬 Research Contributions

1. **Quantum Feature Engineering**: Novel approaches to encode network features in quantum states
2. **Hybrid Architectures**: Classical-quantum hybrid models for cybersecurity
3. **Performance Analysis**: Systematic comparison of quantum vs classical approaches
4. **Practical Implementation**: Real-world application of quantum computing in security

## 🚀 Future Work

- **Larger Quantum Circuits**: Scale to more qubits and complex architectures
- **Real Quantum Hardware**: Deployment on actual quantum computers
- **Advanced Quantum Algorithms**: Implement quantum-specific algorithms (VQE, QAOA)
- **Multi-class Classification**: Extend to multiple attack types
- **Real-time Processing**: Stream processing capabilities for live network monitoring

## 📚 References

- NF-UNSW-NB15 Dataset: Network intrusion detection dataset
- TensorFlow Quantum Documentation
- Cirq Quantum Computing Framework
- Quantum Machine Learning Literature

---

**Note**: This project demonstrates the application of quantum computing principles to cybersecurity, exploring the potential advantages of quantum neural networks in network intrusion detection. The implementation uses state-of-the-art quantum machine learning frameworks while maintaining practical applicability to real-world security challenges. 