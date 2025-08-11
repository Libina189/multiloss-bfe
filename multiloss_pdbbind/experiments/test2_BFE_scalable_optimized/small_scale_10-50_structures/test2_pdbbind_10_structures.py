#!/usr/bin/env python3
"""
Test2 PDBbind 10 Structures - PGGCN Model Training Script

This script implements a Physics-Guided Graph Convolutional Network (PGGCN) model
for predicting binding free energies using a subset of 10 structures from PDBbind dataset.
The model incorporates both empirical and physics-based loss functions.

Converted from Jupyter notebook: Test2_PDBbind_10_structures.ipynb
"""

# ============================================================================
# IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import os
import conda_installer

# Set CUDA environment variables (must come BEFORE importing TensorFlow)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/nvidia/cuda/cuda-11.6'

import pandas as pd
import tensorflow as tf
import numpy as np
from rdkit import Chem
from deepchem.feat.graph_features import atom_features as get_atom_features
import rdkit
import pickle
import copy
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import csv
import time
import sys
import importlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras.backend as K
from tensorflow.keras import regularizers, constraints, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data():
    """Load and preprocess the PDBbind dataset."""
    print("Loading data...")
    df = pd.read_csv('Datasets/pdbbind_100.csv')
    PDBs = pickle.load(open('Datasets/PDBBind_100.pkl', 'rb'))
    
    # Remove problematic entries
    df = df[df['complex-name'] != '1.00E+66']
    
    # Get the first 10 rows for testing
    first_50_rows = df.head(50)
    #print("First 10 rows:")
    print(first_50_rows)
    df = first_50_rows

    # Filter PDBs to match available keys
    pdb_keys = set(PDBs.keys())
    df_filtered = df[df['complex-name'].isin(pdb_keys)]
    print(f"Filtered dataframe length: {len(df_filtered)}")
    
    # Select relevant columns
    df_final = df_filtered[['complex-name','pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals', 
                           'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
                           'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec', 
                           'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb', 
                           'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf','ddg']]
    
    # Get the list of complex names from df_final
    keys_of_interest = df_final['complex-name'].tolist()
    
    # Filter PDBs dictionary to keep only those complexes
    filtered_PDBs = {k: PDBs[k] for k in keys_of_interest if k in PDBs}
    
    print(f"Filtered PDBs count: {len(filtered_PDBs)}")
    print("Filtered PDBs:", filtered_PDBs)
    
    return df_final, filtered_PDBs


def extract_physics_info(df_final, filtered_PDBs):
    """Extract physics information for each PDB structure."""
    info = []
    for pdb in list(filtered_PDBs.keys()):
        physics_data = df_final[df_final['complex-name'] == pdb][['pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals', 
                                                                  'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
                                                                  'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec', 
                                                                  'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb', 
                                                                  'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf']].to_numpy()[0]
        info.append(physics_data)
    return info


# ============================================================================
# FEATURIZATION
# ============================================================================

# Import custom layers and atom features
import models.layers_update_mobley as layers
from models.dcFeaturizer import atom_features as get_atom_features
importlib.reload(layers)


def featurize(molecule, info):
    """
    Featurize a molecule with atom features and physics information.
    
    Args:
        molecule: RDKit molecule object
        info: Physics information array
        
    Returns:
        numpy array of concatenated atom features and physics info
    """
    atom_features = []
    for atom in molecule.GetAtoms():
        new_feature = get_atom_features(atom).tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]
        
        # Add neighbor information (up to 2 neighbors)
        for neighbor in atom.GetNeighbors()[:2]:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [neighbor_idx]
        # Pad with -1 if fewer than 2 neighbors
        for i in range(2 - len(atom.GetNeighbors())):
            new_feature += [-1]

        atom_features.append(np.concatenate([new_feature, info], 0))
    return np.array(atom_features)


def prepare_features(df_final, filtered_PDBs, info):
    """Prepare feature matrices X and target values y."""
    print("Preparing features...")
    X = []
    y = []
    for i, pdb in enumerate(list(filtered_PDBs.keys())):
        print(f"Processing PDB {i+1}/{len(filtered_PDBs)}: {pdb}")
        X.append(featurize(filtered_PDBs[pdb], info[i]))
        y.append(df_final[df_final['complex-name'] == pdb]['ddg'].to_numpy()[0])
    
    # Find max number of atoms in the dataset for padding
    t1 = []
    t2 = []
    for i in range(len(X)):
        shape = X[i].shape
        t1.append(shape[0])
        t2.append(shape[1])
    max_atoms, max_features = max(t1), max(t2)
    print(f"Max atoms: {max_atoms}, Max features: {max_features}")
    
    return X, y


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class PGGCNModel(tf.keras.Model):
    """Physics-Guided Graph Convolutional Network Model."""
    
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=1024, 
                 l2=1e-2, dropout_rate=0.4, maxnorm=3.0):
        super().__init__()
        self.ruleGraphConvLayer = layers.RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.ruleGraphConvLayer.combination_rules = []
        self.conv = layers.ConvLayer(c_out_channel, r_out_channel)
        
        # Dense layers with regularization
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', name='dense1', 
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        self.dense5 = tf.keras.layers.Dense(16, activation='relu', name='dense2', 
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        self.dense6 = tf.keras.layers.Dense(1, name='dense6', 
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        
        # Physics-informed dense layer with custom initialization
        self.dense7 = tf.keras.layers.Dense(1, name='dense7',
                                           kernel_initializer=tf.keras.initializers.Constant([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]),
                                           bias_initializer=tf.keras.initializers.Zeros(),
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))

    def addRule(self, rule, start_index, end_index=None):
        """Add a combination rule to the graph convolution layer."""
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)
        
    def set_input_shapes(self, i_s):
        """Set input shapes for variable-sized molecules."""
        self.i_s = i_s

    def call(self, inputs, training=False):
        """Forward pass of the model."""
        print("Inside call")
        physics_info = inputs[:, 0, 38:] 
        x_a = []
        for i in range(len(self.i_s)):
            x_a.append(inputs[i][:self.i_s[i], :38])
        
        x = self.ruleGraphConvLayer(x_a)
        x = self.conv(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense5(x)
        x = self.dropout2(x, training=training)
        model_var = self.dense6(x)
        
        # Merge model prediction with physics information
        merged = tf.concat([model_var, physics_info], axis=1)
        out = self.dense7(merged)
        
        return tf.concat([out, physics_info], axis=1)


class LossComponentsCallback(tf.keras.callbacks.Callback):
    """Callback to track different loss components during training."""
    
    def __init__(self, model_instance):
        super().__init__()
        self.empirical_losses = []
        self.physical_losses = []
        self.total_losses = []
        self.learning_rates = []
        self.model = model_instance
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch to log metrics."""
        logs = logs or {}
        # Store the total loss
        self.total_losses.append(logs.get('loss'))
        
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        
        self.learning_rates.append(float(tf.keras.backend.get_value(lr)))


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error with additional term."""
    return K.sqrt(K.mean(K.square(y_pred[0] - y_true))) + K.abs(1 / K.mean(.2 + y_pred[1]))


def pure_rmse(y_true, y_pred):
    """Pure root mean squared error."""
    y_true_flat = tf.reshape(y_true, [-1])
    return K.sqrt(K.mean(K.square(y_pred - y_true_flat)))


def physical_consistency_loss(y_true, y_pred, physics_info):
    """
    Physics-based consistency loss function.
    
    Calculates the difference between predicted binding affinity and 
    physics-based calculation using energy components.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dG_pred = y_pred
    y_true = tf.reshape(y_true, (-1, 1))

    # Extract energy components from physics_info
    host = tf.gather(physics_info, [0, 3, 6, 9, 12], axis=1)      # Host energy terms
    guest = tf.gather(physics_info, [1, 4, 7, 10, 13], axis=1)   # Guest energy terms
    complex_ = tf.gather(physics_info, [2, 5, 8, 11, 14], axis=1) # Complex energy terms

    # Calculate ΔG based on physics: ΔG = ΔGcomplex - (ΔGhost + ΔGguest)
    dG_physics = tf.reduce_sum(complex_, axis=1, keepdims=True) - (
        tf.reduce_sum(host, axis=1, keepdims=True) + 
        tf.reduce_sum(guest, axis=1, keepdims=True)
    )
    
    phy_loss = K.sqrt(K.mean(K.square(dG_pred - dG_physics)))
    return phy_loss


def combined_loss(physics_hyperparam=0.0003):
    """
    Combined loss function with empirical and physics components.
    
    Args:
        physics_hyperparam: Weight for physics loss component
        
    Returns:
        Loss function that combines empirical RMSE and physics consistency
    """
    def loss_function(y_true, y_pred):
        # Extract prediction and physics info
        prediction = y_pred[:, 0]
        physics_info = y_pred[:, 1:16]  # Assuming 15 physical features
        
        # Calculate individual loss components
        empirical_loss = pure_rmse(y_true, prediction)
        physics_loss = physical_consistency_loss(y_true, prediction, physics_info)
        
        # Combine losses with weights
        total_loss = empirical_loss + (physics_hyperparam * physics_loss)
        
        return total_loss
    
    return loss_function


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def pad_sequences(X, max_length=12000):
    """Pad sequences to ensure uniform input size."""
    for i in range(len(X)):
        if X[i].shape[0] < max_length:
            padding_size = max_length - X[i].shape[0]
            padding = np.zeros([padding_size, X[i].shape[1]])
            X[i] = np.concatenate([X[i], padding], 0)
    return np.array(X)


def plot_training_results(loss_tracker):
    """Plot training loss over epochs."""
    if not loss_tracker.total_losses:
        print("No training loss data to plot.")
        return
    
    try:
        plt.figure(figsize=(12, 8))
        
        epoch_length = range(1, len(loss_tracker.total_losses) + 1)
        
        # Total loss
        plt.subplot(2, 2, 1)
        plt.plot(epoch_length, loss_tracker.total_losses, 'b-', label='Total Loss', linewidth=2)
        plt.title('Total Loss Over Epochs', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add learning rate plot if available
        if loss_tracker.learning_rates:
            plt.subplot(2, 2, 2)
            plt.plot(epoch_length, loss_tracker.learning_rates, 'r-', label='Learning Rate', linewidth=2)
            plt.title('Learning Rate Over Epochs', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'training_loss_plot.png'")
        
        # Try to show the plot
        plt.show()
        
        # Also print the loss values for verification
        print("\nTraining Loss Values:")
        for i, loss in enumerate(loss_tracker.total_losses):
            print(f"Epoch {i+1}: {loss:.6f}")
            
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("Printing loss values instead:")
        for i, loss in enumerate(loss_tracker.total_losses):
            print(f"Epoch {i+1}: {loss:.6f}")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function."""
    print("Starting PGGCN training script...")
    
    # Load and preprocess data
    df_final, filtered_PDBs = load_data()
    info = extract_physics_info(df_final, filtered_PDBs)
    X, y = prepare_features(df_final, filtered_PDBs, info)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    print(f"Training set: {len(X_train)}, Testing set: {len(X_test)}")
    
    # Training hyperparameters
    physics_hyperparam = [0.000002]
    epochs = [100]
    
    # Learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=5000,
        decay_rate=0.85,
        staircase=True
    )
    
    # Initialize result tracking
    y_differences = []
    total_losses = []
    empirical_losses = []
    physics_losses = []
    all_results = []
    
    start = time.time()
    
    # Training loop
    for epoch in epochs:
        for physics_weight in physics_hyperparam:
            print("---------- Hyperparameter combinations ------------")
            print(f"Epoch: {epoch}, physics_weight: {physics_weight}")
            
            # Initialize model
            m = PGGCNModel()
            m.addRule("sum", 0, 32)
            m.addRule("multiply", 32, 33)
            m.addRule("distance", 33, 36)
            
            # Compile model
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            m.compile(loss=combined_loss(physics_weight), optimizer=opt)
            
            # Prepare training data
            input_shapes = [np.array(X_train[i]).shape[0] for i in range(len(X_train))]
            m.set_input_shapes(input_shapes)
            
            X_train_padded = pad_sequences(copy.deepcopy(X_train))
            y_train_array = np.array(y_train)
            
            # Initialize callbacks
            loss_tracker = LossComponentsCallback(m)
            early_stopping = EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1
            )
            
            # Train model
            print("Starting training...")
            hist = m.fit(X_train_padded, y_train_array, 
                        epochs=epoch, 
                        batch_size=len(X_train_padded),
                        callbacks=[early_stopping, loss_tracker])
            
            # Prepare test data
            input_shapes_test = [np.array(X_test[i]).shape[0] for i in range(len(X_test))]
            m.set_input_shapes(input_shapes_test)
            
            X_test_padded = pad_sequences(copy.deepcopy(X_test))
            y_test_array = np.array(y_test)
            
            # Make predictions
            print("Making predictions...")
            y_pred_test = m.predict(X_test_padded)
            y_pred_test = np.array(y_pred_test[:, 0])
            
            # Calculate metrics
            y_difference = np.mean(np.abs(np.abs(y_test_array) - np.abs(y_pred_test)))
            eval_loss = m.evaluate(X_test_padded, y_test_array)
            
            print(f"Mean absolute difference between y_true & y_pred: {y_difference}")
            
            final_train_loss = loss_tracker.total_losses[-1] if loss_tracker.total_losses else None
            
            # Plot results
            plot_training_results(loss_tracker)
    
    end = time.time()
    print(f"Number of epochs: {epoch}, Runtime: {(end - start)/60:.2f} minutes")


if __name__ == "__main__":
    main()
