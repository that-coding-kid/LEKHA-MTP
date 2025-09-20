import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Training data (29 epochs)
training_data = [
    [0.08430940577918018, 0.08775857263891851, 0.12001434382305878, 0.09578569455012416, 0.09025732302703486], # Epoch 1
    [0.0889307304528732, 0.09131308739280582, 0.11520319063960156, 0.09035305882900643, 0.0862329824629036], # Epoch 2
    [0.08326573789688042, 0.08639502511880431, 0.11620018148399305, 0.08748739286026112, 0.08326318785683123], # Epoch 3
    [0.08107793027844538, 0.08249891915722447, 0.11285614388315669, 0.0829678865722155, 0.08046011293906467], # Epoch 4
    [0.07845491630937136, 0.07928531784935268, 0.1102506776452874, 0.07735702602446565, 0.07387542332770895], # Epoch 5
    [0.07679905013959971, 0.07721200274516887, 0.10764747763608257, 0.06983137938319194, 0.06886582427998375], # Epoch 6
    [0.07334671751627654, 0.07331420209101958, 0.09819585288465428, 0.06550535660628322, 0.06302062004775942], # Epoch 7
    [0.07052467733658699, 0.07079222087439313, 0.0969439559949637, 0.061111516897444776, 0.059177556868192295], # Epoch 8
    [0.06978311186640335, 0.06762948045030662, 0.09392835174325892, 0.05768902168974732, 0.05753639241408147], # Epoch 9
    [0.06505779375314996, 0.06499517921209942, 0.08882421145663373, 0.054520669336746506, 0.054725766463577075], # Epoch 10
    [0.066101780028878, 0.06466039170664109, 0.08455824822055884, 0.05240961842659664, 0.051526104885074635], # Epoch 11
    [0.06266204791625007, 0.06068178295461826, 0.08455002348829566, 0.049829644638827456, 0.04925705599625991], # Epoch 12
    [0.06025931419743138, 0.058763770205708665, 0.08090261312436874, 0.049862466085973695, 0.04837163435923414], # Epoch 13
    [0.058634585888121205, 0.05748292489037181, 0.0801363120200395, 0.04873298310041057, 0.04699695716144482], # Epoch 14
    [0.05659988201826207, 0.05709488727443254, 0.08413539277941946, 0.04847155521308009, 0.04670747263810689], # Epoch 15
    [0.055411721891717246, 0.05489590902226328, 0.07602244253945316, 0.04618000455774896, 0.04595854107712953], # Epoch 16
    [0.05511597135146821, 0.055417945445757635, 0.07469255509484861, 0.04648624740637107, 0.04709088360560209], # Epoch 17
    [0.054910138514586, 0.05339758512140292, 0.07519748164134268, 0.0475563129195973, 0.044647013774456186], # Epoch 18
    [0.053594219751298536, 0.05522613723278754, 0.07226765378694112, 0.045917581098812735, 0.04386160685501625], # Epoch 19
    [0.052687983828794506, 0.052475286796503526, 0.07152342528312211, 0.04547259581687191, 0.0441970007699413], # Epoch 20
    [0.052614050890344766, 0.05168738425783088, 0.0724510953649069, 0.044379687495571814, 0.04279998886736676], # Epoch 21
    [0.052436864020824835, 0.05075385072869331, 0.0723281862747958, 0.04367624052619792, 0.043460473321056126], # Epoch 22
    [0.05121985950640872, 0.051124478134441736, 0.06863285945041207, 0.04461434288599666, 0.04344494904287331], # Epoch 23
    [0.05079811922813761, 0.052732317799288275, 0.07277761645077134, 0.04297468453321493, 0.04221106344693155], # Epoch 24
    [0.05175141518831847, 0.05125447613325842, 0.06910431119893716, 0.04351742635027835, 0.041959899894626715], # Epoch 25
    [0.05048290283565488, 0.05042090651213671, 0.06791476140008719, 0.04315448272636016, 0.04122069435168417], # Epoch 26
    [0.05032979905429929, 0.049196965136806006, 0.06899096154818213, 0.043599660244832326, 0.04174978599427019], # Epoch 27
    [0.05031931791083542, 0.05028451153681343, 0.07402069709905001, 0.04248723579705986, 0.041229757038693436], # Epoch 28
    [0.04967424180105686, 0.049378782451732925, 0.06657214068220367, 0.04181411899875001, 0.041783362212803786] # Epoch 29
]

# Validation data (29 epochs)
validation_data = [
    [0.09876662695647351, 0.09684707728281085, 0.120044452770214, 0.13760875734234496, 0.10925718845932611], # Epoch 1
    [0.09535847226756491, 0.09920183293122266, 0.11971056018956006, 0.1378150373763804, 0.1079683502570593], # Epoch 2
    [0.09350888372864574, 0.0947781195198851, 0.12105983711912163, 0.1401472053091441, 0.11123178809482072], # Epoch 3
    [0.0943801920428606, 0.09498909568147999, 0.12102731140995664, 0.13954197911412589, 0.11971481646677214], # Epoch 4
    [0.09584973845630884, 0.09777584093223725, 0.12173275253735483, 0.14091809860630228, 0.10919500667867917], # Epoch 5
    [0.095598529891244, 0.09766586008481681, 0.12185944487074656, 0.14079625068032847, 0.11445743404328823], # Epoch 6
    [0.09954612713772804, 0.09805794747912192, 0.12243348938812103, 0.14403729420155287, 0.11466403817757964], # Epoch 7
    [0.09801041925259467, 0.10165836759344009, 0.12692404285605466, 0.14752471673169307, 0.11381641199945339], # Epoch 8
    [0.1001514885241964, 0.10060954118047707, 0.12640016991645098, 0.1444809113995039, 0.12581359428752745], # Epoch 9
    [0.10011870588641614, 0.10121007721006338, 0.13037464558146894, 0.15085500881208905, 0.11963275038371128], # Epoch 10
    [0.10626300941554032, 0.10510831236440156, 0.12773733735749765, 0.14779633883985557, 0.11884466312559587], # Epoch 11
    [0.1052305031501289, 0.10351034563167819, 0.12775810385522032, 0.14831954520195723, 0.118813019312386], # Epoch 12
    [0.10206315392029605, 0.10309309376836089, 0.12654616570632374, 0.14825518264634802, 0.1183668553629624], # Epoch 13
    [0.10185756065350558, 0.10445246393127101, 0.12911631190218031, 0.1495565021510369, 0.1224782498154257], # Epoch 14
    [0.10205249190663121, 0.1034459674098928, 0.12909761545181805, 0.15152531986989612, 0.1199344485565754], # Epoch 15
    [0.10436000400555454, 0.10467232540915054, 0.13640679761634342, 0.15041702673105256, 0.1224829142447561], # Epoch 16
    [0.10632478996246521, 0.10864732373738661, 0.1322669830239777, 0.15005177435731248, 0.125169063885031], # Epoch 17
    [0.10443012436319675, 0.1071050595741586, 0.13035037348579084, 0.15182181105150708, 0.122759456829434], # Epoch 18
    [0.10606243893770236, 0.10959070567540559, 0.1317097882773461, 0.16159063674110388, 0.12591677009394125], # Epoch 19
    [0.10456974827684462, 0.1050547758682764, 0.13138125367861772, 0.15010774448247893, 0.12284946837462485], # Epoch 20
    [0.10641214111819863, 0.10922087867012513, 0.14778789897848452, 0.14926224704166607, 0.12230516408037927], # Epoch 21
    [0.10540639858559839, 0.1072292638543461, 0.1320643862709403, 0.15649948921054602, 0.12499795970506966], # Epoch 22
    [0.10575967749381172, 0.10673860961937212, 0.13651245687755623, 0.15091444325766393, 0.12566431117842772], # Epoch 23
    [0.10448231716041587, 0.10700988338794559, 0.13639137388340064, 0.15142081355276918, 0.12429019143538815], # Epoch 24
    [0.10650192134614501, 0.10925467844520297, 0.13816011931547628, 0.15239584272993462, 0.12132241277556334], # Epoch 25
    [0.11040363500693015, 0.10903372926571007, 0.1385982896733497, 0.1507414397949885, 0.13020185843509222], # Epoch 26
    [0.10536998771463654, 0.11215511840834681, 0.13232586762335682, 0.15054133823806687, 0.12323804668683026], # Epoch 27
    [0.10856888998698976, 0.1101905844906079, 0.13634105282835662, 0.1536347845768822, 0.12391129984254283], # Epoch 28
    [0.10966666441942964, 0.10985167502492134, 0.13663771568930574, 0.15297623906683708, 0.12261839373968542] # Epoch 29
]

# Model names and colors
model_names = ['Helpfulness', 'Correctness', 'Coherence', 'Complexity', 'Verbosity']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
epochs = list(range(1, 30))

# Convert to numpy arrays for easier manipulation
train_losses = np.array(training_data)
val_losses = np.array(validation_data)

def create_individual_plots():
    """Create individual plots for each model"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        ax = axes[i]
        
        # Plot training and validation curves
        ax.plot(epochs, train_losses[:, i], 'o-', color=color, linewidth=2, 
                markersize=4, label=f'Training', alpha=0.8)
        ax.plot(epochs, val_losses[:, i], 's--', color=color, linewidth=2, 
                markersize=4, label=f'Validation', alpha=0.8)
        
        ax.set_title(f'{model_name} Model Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add statistics text box
        final_train = train_losses[-1, i]
        final_val = val_losses[-1, i]
        best_train = np.min(train_losses[:, i])
        best_val = np.min(val_losses[:, i])
        
        stats_text = f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}\n' \
                    f'Best Train: {best_train:.4f}\nBest Val: {best_val:.4f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Remove the empty subplot
    axes[5].remove()
    
    plt.tight_layout()
    return fig

def create_combined_plot():
    """Create combined plot showing all models together"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training losses
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        ax1.plot(epochs, train_losses[:, i], 'o-', color=color, linewidth=2, 
                markersize=3, label=model_name, alpha=0.8)
    
    ax1.set_title('Training Loss Curves', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Validation losses
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        ax2.plot(epochs, val_losses[:, i], 's--', color=color, linewidth=2, 
                markersize=3, label=model_name, alpha=0.8)
    
    ax2.set_title('Validation Loss Curves', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def create_summary_plot():
    """Create summary plot with statistics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Final losses comparison
    final_train_losses = train_losses[-1, :]
    final_val_losses = val_losses[-1, :]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, final_train_losses, width, label='Training', 
                   color=colors, alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, final_val_losses, width, label='Validation',
                   color=colors, alpha=0.6)
    
    ax1.set_title('Final Loss Comparison (Epoch 29)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Best losses comparison
    best_train_losses = np.min(train_losses, axis=0)
    best_val_losses = np.min(val_losses, axis=0)
    
    bars3 = ax2.bar(x_pos - width/2, best_train_losses, width, label='Training', 
                   color=colors, alpha=0.8)
    bars4 = ax2.bar(x_pos + width/2, best_val_losses, width, label='Validation',
                   color=colors, alpha=0.6)
    
    ax2.set_title('Best Loss Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Models', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Training progress (loss reduction)
    initial_train = train_losses[0, :]
    final_train = train_losses[-1, :]
    train_reduction = ((initial_train - final_train) / initial_train) * 100
    
    bars5 = ax3.bar(model_names, train_reduction, color=colors, alpha=0.8)
    ax3.set_title('Training Loss Reduction (%)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Models', fontsize=12)
    ax3.set_ylabel('Reduction (%)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for bar in bars5:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 4. Loss curves for best performing model
    best_model_idx = np.argmin(final_val_losses)
    best_model_name = model_names[best_model_idx]
    best_color = colors[best_model_idx]
    
    ax4.plot(epochs, train_losses[:, best_model_idx], 'o-', color=best_color, 
            linewidth=3, markersize=4, label=f'{best_model_name} - Training', alpha=0.8)
    ax4.plot(epochs, val_losses[:, best_model_idx], 's--', color=best_color, 
            linewidth=3, markersize=4, label=f'{best_model_name} - Validation', alpha=0.8)
    
    ax4.set_title(f'Best Model: {best_model_name}', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_summary_statistics():
    """Print detailed summary statistics"""
    print("\n" + "="*80)
    print("TRAINING SUMMARY STATISTICS")
    print("="*80)
    
    for i, model_name in enumerate(model_names):
        print(f"\n{model_name.upper()} MODEL:")
        print("-" * 40)
        print(f"Initial Training Loss: {train_losses[0, i]:.6f}")
        print(f"Final Training Loss:   {train_losses[-1, i]:.6f}")
        print(f"Best Training Loss:    {np.min(train_losses[:, i]):.6f}")
        print(f"Training Improvement:  {((train_losses[0, i] - train_losses[-1, i])/train_losses[0, i]*100):.1f}%")
        
        print(f"Initial Validation Loss: {val_losses[0, i]:.6f}")
        print(f"Final Validation Loss:   {val_losses[-1, i]:.6f}")
        print(f"Best Validation Loss:    {np.min(val_losses[:, i]):.6f}")
        print(f"Best Val Epoch:          {np.argmin(val_losses[:, i]) + 1}")
    
    print(f"\n{'='*80}")
    print("OVERALL RANKING (by final validation loss):")
    print("="*80)
    final_val = val_losses[-1, :]
    sorted_indices = np.argsort(final_val)
    
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"{rank}. {model_names[idx]}: {final_val[idx]:.6f}")

# Create and save plots
if __name__ == "__main__":
    # Create plots
    print("Creating individual model plots...")
    fig1 = create_individual_plots()
    
    print("Creating combined plots...")
    fig2 = create_combined_plot()
    
    print("Creating summary plots...")
    fig3 = create_summary_plot()
    
    # Save plots
    print("Saving plots...")
    fig1.savefig('individual_model_losses.png', dpi=300, bbox_inches='tight')
    fig1.savefig('individual_model_losses.pdf', bbox_inches='tight')
    
    fig2.savefig('combined_loss_curves.png', dpi=300, bbox_inches='tight')
    fig2.savefig('combined_loss_curves.pdf', bbox_inches='tight')
    
    fig3.savefig('loss_summary_analysis.png', dpi=300, bbox_inches='tight')
    fig3.savefig('loss_summary_analysis.pdf', bbox_inches='tight')
    
    # Print summary statistics
    print_summary_statistics()
    
    print("\nPlots saved successfully!")
    print("Files created:")
    print("- individual_model_losses.png/pdf")
    print("- combined_loss_curves.png/pdf") 
    print("- loss_summary_analysis.png/pdf")
    
    # Show plots (optional - comment out if running headless)
    plt.show()