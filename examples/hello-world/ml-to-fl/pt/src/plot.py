import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_distribution_org(df: pd.DataFrame, save_path="./class_distribution.png"):
    """
    Visualize the class distribution in the original dataset with counts and percentages.
    Parameters:
    -----------
    df : pandas DataFrame
        The original dataset containing a 'Class' column where 0 represents non-fraud and 1 represents fraud
    save_path : str
        Path where the figure will be saved
    Returns:
    --------
    None (saves the plot to file)
    """
    y = df['Class']
    
    class_0_count = np.sum(y == 0)
    class_1_count = np.sum(y == 1)
    total = len(y)
    class_0_percentage = class_0_count / total * 100
    class_1_percentage = class_1_count / total * 100
    
    plt.figure(figsize=(12, 8))
    
    colors = ["#00008B", "#FFA500"] 
    
    ax = sns.countplot(x='Class', hue='Class', data=df, palette=colors, legend=False)
    
    for i, p in enumerate(ax.patches):
        count = class_0_count if i == 0 else class_1_count
        percentage = class_0_percentage if i == 0 else class_1_percentage
        
        text_color = 'white' if i == 0 else 'red'
        count_display = class_0_count if i == 0 else class_1_count
        
        ylim = ax.get_ylim()
        y_position = ylim[1]/2  
        
        ax.text(p.get_x() + p.get_width()/2, y_position,
                f'{count_display:,}\n({percentage:.2f}%)', ha="center", va="center", fontsize=14,
                color=text_color, fontweight='bold')
    
    plt.title('Class Distribution in Original Dataset\n(0: No Fraud || 1: Fraud)', fontsize=14, pad=30, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks([0, 1], ['No Fraud (0)', 'Fraud (1)'])
    
    imbalance_ratio = class_0_count / class_1_count if class_1_count > 0 else float('inf')
    stats_text = (
        f"Dataset Statistics:\n"
        f"Total samples: {total:,}\n"
        f"No Fraud (0): {class_0_count:,} ({class_0_percentage:.2f}%)\n"
        f"Fraud (1): {class_1_count:,} ({class_1_percentage:.2f}%)\n"
        f"Imbalance ratio: {imbalance_ratio:.1f}:1"
    )
    
    plt.text(0.98, 0.90, stats_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.8'))
    
    current_ylim = plt.ylim()
    plt.ylim(0, current_ylim[1] * 1.1) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_before_after_smote(y_train_before, y_train_after, save_path="./smote_comparison.png"):
    """
    Visualize the class distribution before and after applying SMOTE.
    
    Parameters:
    -----------
    y_train_before : array-like
        Class labels before applying SMOTE where 0 represents non-fraud and 1 represents fraud
    y_train_after : array-like
        Class labels after applying SMOTE where 0 represents non-fraud and 1 represents fraud
    save_path : str
        Path where the figure will be saved
    
    Returns:
    --------
    None (saves the plot to file)
    """
    # Convert inputs to numpy arrays if they aren't already
    y_before = np.array(y_train_before)
    y_after = np.array(y_train_after)
    
    # Calculate counts and percentages for before SMOTE
    before_class_0_count = np.sum(y_before == 0)
    before_class_1_count = np.sum(y_before == 1)
    before_total = len(y_before)
    before_class_0_percentage = before_class_0_count / before_total * 100
    before_class_1_percentage = before_class_1_count / before_total * 100
    
    # Calculate counts and percentages for after SMOTE
    after_class_0_count = np.sum(y_after == 0)
    after_class_1_count = np.sum(y_after == 1)
    after_total = len(y_after)
    after_class_0_percentage = after_class_0_count / after_total * 100
    after_class_1_percentage = after_class_1_count / after_total * 100
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define colors for non-fraud and fraud - dark blue and orange
    colors = ["#00008B", "#FFA500"]  # Dark blue, Orange
    
    # Create data frames for plotting
    df_before = pd.DataFrame({'Class': np.concatenate([np.zeros(before_class_0_count), np.ones(before_class_1_count)])})
    df_after = pd.DataFrame({'Class': np.concatenate([np.zeros(after_class_0_count), np.ones(after_class_1_count)])})
    
    # ---------- FIRST SUBPLOT: BEFORE SMOTE ----------
    sns.countplot(x='Class', hue='Class', data=df_before, palette=colors, legend=False, ax=ax1)
    
    # Get the axis limits for consistent positioning
    y_limit_before = ax1.get_ylim()[1]
    
    # Add text in the middle of each bar
    for i, p in enumerate(ax1.patches):
        count = before_class_0_count if i == 0 else before_class_1_count
        percentage = before_class_0_percentage if i == 0 else before_class_1_percentage
        text_color = 'white' if i == 0 else 'red'
        
        # Position text at the middle of the figure height
        ax1.text(p.get_x() + p.get_width()/2, y_limit_before/2,
                f'{count:,}\n({percentage:.2f}%)', ha="center", va="center", 
                fontsize=14, color=text_color, fontweight='bold')
    
    # Add title and labels for first subplot
    ax1.set_title('Class Distribution Before SMOTE\n(0: No Fraud || 1: Fraud)', fontsize=14, fontweight='bold', pad=30)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['No Fraud (0)', 'Fraud (1)'])
    
    # Add a text box with summary statistics for before SMOTE
    before_imbalance_ratio = before_class_0_count / before_class_1_count if before_class_1_count > 0 else float('inf')
    before_stats_text = (
        f"Before SMOTE Statistics:\n"
        f"Total samples: {before_total:,}\n"
        f"No Fraud (0): {before_class_0_count:,} ({before_class_0_percentage:.2f}%)\n"
        f"Fraud (1): {before_class_1_count:,} ({before_class_1_percentage:.2f}%)\n"
        f"Imbalance ratio: {before_imbalance_ratio:.1f}:1"
    )
    
    ax1.text(0.98, 0.90, before_stats_text,
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.8'))
    
    # ---------- SECOND SUBPLOT: AFTER SMOTE ----------
    sns.countplot(x='Class', hue='Class', data=df_after, palette=colors, legend=False, ax=ax2)
    
    # Get the axis limits for consistent positioning
    y_limit_after = ax2.get_ylim()[1]
    
    # Add text in the middle of each bar
    for i, p in enumerate(ax2.patches):
        count = after_class_0_count if i == 0 else after_class_1_count
        percentage = after_class_0_percentage if i == 0 else after_class_1_percentage
        text_color = 'white' if i == 0 else 'red'
        
        # Position text at the middle of the figure height
        ax2.text(p.get_x() + p.get_width()/2, y_limit_after/2,
                f'{count:,}\n({percentage:.2f}%)', ha="center", va="center", 
                fontsize=14, color=text_color, fontweight='bold')
    
    # Add title and labels for second subplot
    ax2.set_title('Class Distribution After SMOTE\n(0: No Fraud || 1: Fraud)', fontsize=14, fontweight='bold', pad=30)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['No Fraud (0)', 'Fraud (1)'])
    
    # Add a text box with summary statistics for after SMOTE
    after_imbalance_ratio = after_class_0_count / after_class_1_count if after_class_1_count > 0 else float('inf')
    after_stats_text = (
        f"After SMOTE Statistics:\n"
        f"Total samples: {after_total:,}\n"
        f"No Fraud (0): {after_class_0_count:,} ({after_class_0_percentage:.2f}%)\n"
        f"Fraud (1): {after_class_1_count:,} ({after_class_1_percentage:.2f}%)\n"
        f"Imbalance ratio: {after_imbalance_ratio:.1f}:1"
    )
    
    ax2.text(0.98, 0.90, after_stats_text,
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.8'))
    
    # Add overall title for the figure
    plt.suptitle('Impact of SMOTE on Class Distribution', fontsize=16, fontweight='bold', y=1.05)
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_distribution_train_valid_test(y_train, y_valid, y_test, save_path="./class_distribution_splits.png"):
    """
    Visualize the class distribution across train, validation, and test datasets.
    
    Parameters:
    -----------
    y_train : array-like
        Labels from the training dataset
    y_valid : array-like
        Labels from the validation dataset
    y_test : array-like
        Labels from the test dataset
    save_path : str
        Path where the figure will be saved
        
    Returns:
    --------
    None (saves the plot to file)
    """
    # Calculate counts and percentages
    train_0_count = np.sum(y_train == 0)
    train_1_count = np.sum(y_train == 1)
    train_total = len(y_train)
    train_0_pct = train_0_count / train_total * 100
    train_1_pct = train_1_count / train_total * 100
    
    valid_0_count = np.sum(y_valid == 0)
    valid_1_count = np.sum(y_valid == 1)
    valid_total = len(y_valid)
    valid_0_pct = valid_0_count / valid_total * 100
    valid_1_pct = valid_1_count / valid_total * 100
    
    test_0_count = np.sum(y_test == 0)
    test_1_count = np.sum(y_test == 1)
    test_total = len(y_test)
    test_0_pct = test_0_count / test_total * 100
    test_1_pct = test_1_count / test_total * 100
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create subplots - one with log scale, one without
    plt.subplot(2, 1, 1)
    
    # Setup data for grouped bar chart
    datasets = ['Train', 'Validation', 'Test']
    non_fraud_counts = [train_0_count, valid_0_count, test_0_count]
    fraud_counts = [train_1_count, valid_1_count, test_1_count]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Define colors for non-fraud and fraud
    colors = ["#00008B", "#FFA500"]  # Dark blue, Orange
    
    # Create the bar chart - normal scale
    ax1 = plt.gca()
    bars1 = ax1.bar(x - width/2, non_fraud_counts, width, label='No Fraud (0)', color=colors[0])
    bars2 = ax1.bar(x + width/2, fraud_counts, width, label='Fraud (1)', color=colors[1])
    
    # Add count and percentage labels
    for i, bars in enumerate([bars1, bars2]):
        counts = non_fraud_counts if i == 0 else fraud_counts
        for j, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (counts[j] / [train_total, valid_total, test_total][j]) * 100
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{counts[j]:,}\n({percentage:.2f}%)', 
                    ha='center', va='bottom', fontsize=10)
    
    ax1.set_title('Class Distribution Across Datasets', fontsize=14)
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Create the second subplot with log scale
    plt.subplot(2, 1, 2)
    ax2 = plt.gca()
    bars3 = ax2.bar(x - width/2, non_fraud_counts, width, label='No Fraud (0)', color=colors[0])
    bars4 = ax2.bar(x + width/2, fraud_counts, width, label='Fraud (1)', color=colors[1])
    
    # Add count and percentage labels
    for i, bars in enumerate([bars3, bars4]):
        counts = non_fraud_counts if i == 0 else fraud_counts
        for j, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (counts[j] / [train_total, valid_total, test_total][j]) * 100
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{counts[j]:,}\n({percentage:.2f}%)', 
                    ha='center', va='bottom', fontsize=10)
    
    ax2.set_title('Class Distribution Across Datasets (Log Scale)', fontsize=14)
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('Count (log scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add a summary statistics box in the top right of the figure
    total_samples = train_total + valid_total + test_total
    stats_text = (
        f"Dataset Split Statistics:\n\n"
        f"Train set: {train_total:,} samples ({train_total/total_samples*100:.1f}%)\n"
        f"  - Class 0: {train_0_count:,} ({train_0_pct:.2f}%)\n"
        f"  - Class 1: {train_1_count:,} ({train_1_pct:.2f}%)\n"
        f"  - Ratio: {train_0_count/train_1_count:.1f}:1\n\n"
        f"Validation set: {valid_total:,} samples ({valid_total/total_samples*100:.1f}%)\n"
        f"  - Class 0: {valid_0_count:,} ({valid_0_pct:.2f}%)\n"
        f"  - Class 1: {valid_1_count:,} ({valid_1_pct:.2f}%)\n"
        f"  - Ratio: {valid_0_count/valid_1_count:.1f}:1\n\n"
        f"Test set: {test_total:,} samples ({test_total/total_samples*100:.1f}%)\n"
        f"  - Class 0: {test_0_count:,} ({test_0_pct:.2f}%)\n"
        f"  - Class 1: {test_1_count:,} ({test_1_pct:.2f}%)\n"
        f"  - Ratio: {test_0_count/test_1_count:.1f}:1"
    )
    
    plt.figtext(0.77, 0.97, stats_text, fontsize=10,
                bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.8'),
                verticalalignment='top')
    
    plt.tight_layout(rect=[0, 0, 0.75, 0.97])  # Adjust layout to make room for the text box
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_learning_curve(train_metrics_list, valid_metrics_list, minibatch_loss_list=None, save_path="./learning_curve.png"):
    """
    Create a comprehensive learning curve showing loss and all metrics together for easier analysis.
    
    Parameters:
    -----------
    train_metrics_list : list of dict
        List of training metrics dictionaries
    valid_metrics_list : list of dict
        List of validation metrics dictionaries
    minibatch_loss_list : list, optional
        List of training losses per minibatch
    save_path : str
        Path where the figure will be saved
        
    Returns:
    --------
    None (saves the plot to file)
    """
    plt.figure(figsize=(15, 12))
    
    # Plot loss curve if provided
    if minibatch_loss_list:
        plt.subplot(2, 1, 1)
        plt.plot(minibatch_loss_list, color='#FF6347', alpha=0.6, linewidth=1)
        
        # Add moving average for clarity
        window_size = min(100, len(minibatch_loss_list) // 10)
        if window_size > 0:
            moving_avg = pd.Series(minibatch_loss_list).rolling(window=window_size).mean()
            plt.plot(moving_avg, color='#B22222', linewidth=2, label=f'Moving Avg (window={window_size})')
        
        plt.title('Training Loss per Minibatch')
        plt.xlabel('Minibatch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        if window_size > 0:
            plt.legend()
    
    # Plot all metrics in one graph for comparison
    plot_position = 2 if minibatch_loss_list else 1
    plt.subplot(2, 1, plot_position)
    
    epochs = range(1, len(train_metrics_list) + 1)
    
    # Extract all metrics
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    if 'auc_pr' in valid_metrics_list[0]:
        metrics_to_plot.append('auc_pr')
    
    for metric in metrics_to_plot:
        train_values = [metrics[metric] for metrics in train_metrics_list]
        valid_values = [metrics[metric] for metrics in valid_metrics_list]
        
        # Plot training metrics with dotted lines
        plt.plot(epochs, train_values, '--o', alpha=0.6, label=f'Train {metric.replace("_", " ").title()}')
        
        # Plot validation metrics with solid lines
        plt.plot(epochs, valid_values, '-s', linewidth=2, label=f'Valid {metric.replace("_", " ").title()}')
    
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Mark the best validation F1-score
    best_f1_idx = np.argmax([metrics['f1_score'] for metrics in valid_metrics_list])
    best_f1 = valid_metrics_list[best_f1_idx]['f1_score']
    plt.axvline(x=best_f1_idx+1, color='r', linestyle='--', alpha=0.5)
    plt.text(best_f1_idx+1.1, 50, f'Best F1: {best_f1:.2f}% (Epoch {best_f1_idx+1})', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curve visualization saved to {save_path}")

def plot_confusion_matrices(model, test_loader, threshold=0.9, save_path="./confusion_matrices.png"):
    """
    Generate and save a confusion matrix visualization for model predictions
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    test_loader : DataLoader
        DataLoader containing the test data
    threshold : float
        Probability threshold for binary classification
    save_path : str
        Path where the confusion matrix visualization will be saved
    
    Returns:
    --------
    confusion_matrix : numpy.ndarray
        The computed confusion matrix
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Gather all test data
    x_test = [x for x, y in test_loader]
    x_test_tensor = torch.cat(x_test, dim=0).cuda()
    y_test = [y for x, y in test_loader]
    y_test_tensor = torch.cat(y_test, dim=0)
    y_test_numpy = y_test_tensor.view(-1).cpu().numpy()
    
    # Get model predictions
    with torch.no_grad():
        logits = model(x_test_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        y_pred_numpy = (probabilities >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test_numpy, y_pred_numpy)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Fraud (0)', 'Fraud (1)'],
               yticklabels=['No Fraud (0)', 'Fraud (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Threshold: {threshold})')
    
    # Add accuracy, precision, recall, and F1 score
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.figtext(0.5, 0.01, 
               f"Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% | "
               f"Recall: {recall:.2f}% | F1-Score: {f1:.2f}%",
               ha="center", fontsize=10, 
               bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=0.5'))
    
    # Save the plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix visualization saved to {save_path}")
    
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def plot_aucpr(model, test_loader, device, save_path="./auc_pr.png"):
    """
    Plot Precision-Recall curve and calculate AUC-PR for test data after training a model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained PyTorch model to evaluate.
    test_loader : torch.utils.data.DataLoader
        DataLoader containing test data.
    device : torch.device
        Device to run the model on ('cpu' or 'cuda').
    save_path : str
        Path where the plot will be saved.
        
    Returns:
    --------
    float: The AUC-PR score
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    auc_pr = average_precision_score(all_labels, all_preds)
    baseline = np.mean(all_labels)

    plt.figure(figsize=(8, 6))

    # Fill under the PR curve
    plt.fill_between(recall, precision, alpha=0.3, color='orange')

    # Plot the PR curve line
    plt.plot(recall, precision, color='#00008B', linewidth=1.8)  # dark blue

    # Plot the baseline
    plt.axhline(y=baseline, linestyle='--', color='gray', alpha=0.8, linewidth=1.0)
    plt.text(0.02, baseline + 0.015, f"Baseline, y = {baseline:.2f}",
             color='black', fontsize=10)

    # Center the AUC text
    center_x = np.mean(recall)
    center_y = np.mean(precision)
    plt.text(center_x, center_y, f"PR AUC = {auc_pr:.2f}",
             fontsize=14, weight='bold', color='black')


    # Final plot styling
    plt.title("PR curve", fontsize=16)
    plt.xlabel("Recall/True positive rate", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"AUC-PR: {auc_pr:.3f}, plot saved to {save_path}")
    return auc_pr

def plot_metrics(metrics, fig_name="Training Metrics", save_path="./metrics_plot.png"):
    """
    Plot accuracy, precision, recall, and loss metrics in a single figure with four subplots.
    
    Parameters:
    -----------
    metrics : list of dict
        List of metrics dictionaries, each containing 'accuracy', 'precision', 'recall', and 'loss'
    fig_name : str
        Title for the entire figure
    save_path : str
        Path where the figure will be saved
        
    Returns:
    --------
    None (saves the plot to file)
    """
    # Check if input is empty
    if not metrics:
        print("Error: Empty metrics list provided")
        return
        
    # Create figure with 4 subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(fig_name, fontsize=16)
    
    # Prepare data
    epochs = range(1, len(metrics) + 1)
    
    # Extract metrics
    accuracy = [metric['accuracy'] for metric in metrics]
    precision = [metric['precision'] for metric in metrics]
    recall = [metric['recall'] for metric in metrics]
    
    # Extract loss (if available)
    if 'loss' in metrics[0]:
        loss = [metric['loss'] for metric in metrics]
    else:
        loss = None
        print("Warning: Loss metric not found in the provided data")
    
    # Colors
    line_color = '#00008B'  # Dark blue
    
    # Flatten 2x2 array of subplots for easier indexing
    axs = axs.flatten()
    
    # Accuracy plot (top left)
    axs[0].plot(epochs, accuracy, marker='o', color=line_color)
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].grid(True, alpha=0.3)
    
    # Add best accuracy point
    best_acc_idx = np.argmax(accuracy)
    axs[0].plot(best_acc_idx + 1, accuracy[best_acc_idx], 'ro', markersize=8)
    axs[0].text(best_acc_idx + 1.1, accuracy[best_acc_idx],
              f'Best: {accuracy[best_acc_idx]:.2f}%', fontsize=9)
    
    # Precision plot (top right)
    axs[1].plot(epochs, precision, marker='o', color=line_color)
    axs[1].set_title('Precision')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Precision (%)')
    axs[1].grid(True, alpha=0.3)
    
    # Add best precision point
    best_prec_idx = np.argmax(precision)
    axs[1].plot(best_prec_idx + 1, precision[best_prec_idx], 'ro', markersize=8)
    axs[1].text(best_prec_idx + 1.1, precision[best_prec_idx],
              f'Best: {precision[best_prec_idx]:.2f}%', fontsize=9)
    
    # Recall plot (bottom left)
    axs[2].plot(epochs, recall, marker='o', color=line_color)
    axs[2].set_title('Recall')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Recall (%)')
    axs[2].grid(True, alpha=0.3)
    
    # Add best recall point
    best_rec_idx = np.argmax(recall)
    axs[2].plot(best_rec_idx + 1, recall[best_rec_idx], 'ro', markersize=8)
    axs[2].text(best_rec_idx + 1.1, recall[best_rec_idx],
              f'Best: {recall[best_rec_idx]:.2f}%', fontsize=9)
    
    # Loss plot (bottom right)
    if loss:
        axs[3].plot(epochs, loss, marker='o', color='#8B0000')  # Dark red for loss
        axs[3].set_title('Loss')
        axs[3].set_xlabel('Epoch')
        axs[3].set_ylabel('Loss')
        axs[3].grid(True, alpha=0.3)
        
        # Add best (minimum) loss point
        best_loss_idx = np.argmin(loss)
        axs[3].plot(best_loss_idx + 1, loss[best_loss_idx], 'bo', markersize=8)
        axs[3].text(best_loss_idx + 1.1, loss[best_loss_idx],
                  f'Best: {loss[best_loss_idx]:.4f}', fontsize=9)
    else:
        axs[3].text(0.5, 0.5, 'Loss data not available', 
                  horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics visualization saved to {save_path}")