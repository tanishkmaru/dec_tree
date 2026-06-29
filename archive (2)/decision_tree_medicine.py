# =============================================================================
#  Medicine Recommendation — Decision Tree (Depth-limited & Unlimited)
#
#  DATASET (Kaggle):
#    Disease Symptom Description Dataset — itachi9604
#    https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
#
#  FILES NEEDED (place in same folder as this script):
#    - dataset.csv           (disease + up to 17 symptom columns, 4920 rows)
#    - symptom_severity.csv  (symptom name → severity weight 1-7)
#
#  INSTALL:
#    pip install pandas numpy scikit-learn matplotlib seaborn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# =============================================================================
#  STEP 1 — Load dataset
# =============================================================================

print("=" * 60)
print("  STEP 1 — Loading dataset")
print("=" * 60)

df_raw = pd.read_csv("dataset.csv")
df_sev = pd.read_csv("Symptom-severity.csv")

# Clean column names and disease names
df_raw.columns = df_raw.columns.str.strip()
df_raw['Disease'] = df_raw['Disease'].str.strip()
df_sev.columns  = df_sev.columns.str.strip()
df_sev['Symptom'] = df_sev['Symptom'].str.strip().str.lower().str.replace(' ', '_')

print(f"Raw dataset shape  : {df_raw.shape}")
print(f"Unique diseases    : {df_raw['Disease'].nunique()}")
print(f"Unique severities  : {df_sev['Symptom'].nunique()}")

# =============================================================================
#  STEP 2 — Disease → Medicine mapping
# =============================================================================

DISEASE_TO_MEDICINE = {
    'Fungal infection':                          'Antifungal',
    'Allergy':                                   'Antihistamine',
    'GERD':                                      'Antacid',
    'Chronic cholestasis':                       'Antacid',
    'Drug Reaction':                             'Corticosteroid',
    'Peptic ulcer diseae':                       'Antacid',
    'AIDS':                                      'Antiretroviral',
    'Diabetes ':                                 'Insulin',
    'Gastroenteritis':                           'Antiemetic',
    'Bronchial Asthma':                          'Bronchodilator',
    'Hypertension ':                             'Antihypertensive',
    'Migraine':                                  'Paracetamol',
    'Cervical spondylosis':                      'NSAID',
    'Paralysis (brain hemorrhage)':              'Neuroprotective',
    'Jaundice':                                  'Supportive Care',
    'Malaria':                                   'Antimalarial',
    'Chicken pox':                               'Antiviral',
    'Dengue':                                    'Paracetamol',
    'Typhoid':                                   'Antibiotic',
    'hepatitis A':                               'Supportive Care',
    'Hepatitis B':                               'Antiviral',
    'Hepatitis C':                               'Antiviral',
    'Hepatitis D':                               'Antiviral',
    'Hepatitis E':                               'Supportive Care',
    'Alcoholic hepatitis':                       'Corticosteroid',
    'Tuberculosis':                              'Antibiotic',
    'Common Cold':                               'Antihistamine',
    'Pneumonia':                                 'Antibiotic',
    'Dimorphic hemmorhoids(piles)':              'Laxative',
    'Heart attack':                              'Aspirin',
    'Varicose veins':                            'NSAID',
    'Hypothyroidism':                            'Levothyroxine',
    'Hyperthyroidism':                           'Antithyroid',
    'Hypoglycemia':                              'Dextrose',
    'Osteoarthristis':                           'NSAID',
    'Arthritis':                                 'NSAID',
    '(vertigo) Paroymsal  Positional Vertigo':   'Antihistamine',
    'Acne':                                      'Topical Retinoid',
    'Urinary tract infection':                   'Antibiotic',
    'Psoriasis':                                 'Corticosteroid',
    'Impetigo':                                  'Antibiotic',
}

# =============================================================================
#  STEP 3 — Build one-hot symptom matrix  (keep ALL 4920 rows)
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 3 — Building symptom matrix (all rows)")
print("=" * 60)

symptom_cols = [c for c in df_raw.columns if c.startswith('Symptom_')]
valid_syms   = set(df_sev['Symptom'].values)

# Collect every symptom name that appears in the severity file
all_sym_names = set()
for col in symptom_cols:
    cleaned = df_raw[col].dropna().str.strip().str.lower().str.replace(' ', '_')
    all_sym_names.update(cleaned[cleaned.isin(valid_syms)].values)
all_sym_names = sorted(all_sym_names)

# Build binary matrix row by row (preserves all 4920 rows)
rows = []
for _, row in df_raw.iterrows():
    syms_in_row = set()
    for col in symptom_cols:
        val = str(row[col]).strip().lower().replace(' ', '_') if pd.notna(row[col]) else ''
        if val in valid_syms:
            syms_in_row.add(val)
    entry = {s: int(s in syms_in_row) for s in all_sym_names}
    entry['Disease'] = row['Disease'].strip()
    rows.append(entry)

df = pd.DataFrame(rows)
df['medicine'] = df['Disease'].map(DISEASE_TO_MEDICINE)
df = df.dropna(subset=['medicine'])

ALL_SYMPTOMS = all_sym_names
print(f"Matrix shape       : {df.shape}")
print(f"Symptom features   : {len(ALL_SYMPTOMS)}")
print(f"Unique medicines   : {df['medicine'].nunique()}")
print("\nMedicine distribution:")
print(df['medicine'].value_counts())

# =============================================================================
#  STEP 4 — Train / test split
# =============================================================================

X  = df[ALL_SYMPTOMS].values
le = LabelEncoder()
y  = le.fit_transform(df['medicine'])
class_names = le.classes_

# Drop any class with fewer than 5 samples
counts = pd.Series(y).value_counts()
keep   = counts[counts >= 5].index
mask   = pd.Series(y).isin(keep).values
X, y   = X[mask], y[mask]

le = LabelEncoder()
y  = le.fit_transform(pd.Series(y).map({v: le.classes_[v] if v < len(le.classes_) else v
                                         for v in range(len(le.classes_))}).values
                      if False else df['medicine'][mask].values)
class_names = le.classes_

print(f"\nAfter quality filter : {len(y)} rows, {len(class_names)} medicine classes")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {len(X_train)}  |  Test : {len(X_test)}")

# =============================================================================
#  STEP 5A — Depth-limited trees  (n = 1 … 7)
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 5A — Depth-limited trees (n = 1 … 7)")
print("=" * 60)

depth_results = {}
for n in range(1, 8):
    clf = DecisionTreeClassifier(max_depth=n, criterion='gini', random_state=42)
    clf.fit(X_train, y_train)
    tr = accuracy_score(y_train, clf.predict(X_train))
    te = accuracy_score(y_test,  clf.predict(X_test))
    cv = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
    depth_results[n] = {'train': tr, 'test': te, 'cv': cv,
                        'leaves': clf.get_n_leaves(), 'clf': clf}
    print(f"  depth={n}  train={tr:.2%}  test={te:.2%}  cv={cv:.2%}  leaves={clf.get_n_leaves()}")

best_n      = max(depth_results, key=lambda d: depth_results[d]['cv'])
clf_limited = depth_results[best_n]['clf']
print(f"\n  Best depth (by CV accuracy) : {best_n}")

# =============================================================================
#  STEP 5B — Unlimited tree
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 5B — Unlimited tree")
print("=" * 60)

clf_unlimited = DecisionTreeClassifier(max_depth=None, criterion='gini',
                                       min_samples_leaf=2, random_state=42)
clf_unlimited.fit(X_train, y_train)

tr_u = accuracy_score(y_train, clf_unlimited.predict(X_train))
te_u = accuracy_score(y_test,  clf_unlimited.predict(X_test))
cv_u = cross_val_score(clf_unlimited, X, y, cv=5, scoring='accuracy').mean()
print(f"  Depth   : {clf_unlimited.get_depth()}")
print(f"  Leaves  : {clf_unlimited.get_n_leaves()}")
print(f"  Train   : {tr_u:.2%}  |  Test : {te_u:.2%}  |  CV : {cv_u:.2%}")

# =============================================================================
#  STEP 6 — Reports
# =============================================================================

print("\n" + "=" * 60)
print(f"  CLASSIFICATION REPORT — Depth-limited (n={best_n})")
print("=" * 60)
print(classification_report(y_test, clf_limited.predict(X_test),
                             target_names=class_names, zero_division=0))

print("\n" + "=" * 60)
print("  CLASSIFICATION REPORT — Unlimited")
print("=" * 60)
print(classification_report(y_test, clf_unlimited.predict(X_test),
                             target_names=class_names, zero_division=0))

print("\n" + "=" * 60)
print(f"  DECISION RULES — Depth-limited (n={best_n})")
print("=" * 60)
print(export_text(clf_limited, feature_names=list(ALL_SYMPTOMS)))

# =============================================================================
#  STEP 7 — Plots
# =============================================================================

# 7a — Accuracy vs depth
depths = list(depth_results.keys())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(depths, [depth_results[d]['train'] for d in depths], 'o-',  color='#2a78d6', lw=2, label='Train')
axes[0].plot(depths, [depth_results[d]['test']  for d in depths], 's--', color='#e34948', lw=2, label='Test')
axes[0].plot(depths, [depth_results[d]['cv']    for d in depths], '^:',  color='#1baf7a', lw=2, label='CV')
axes[0].axvline(best_n, color='#eda100', ls='--', lw=1.5, label=f'Best n={best_n}')
axes[0].set_title('Accuracy vs Tree Depth'); axes[0].set_xlabel('Max Depth'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].bar(depths, [depth_results[d]['leaves'] for d in depths], color='#2a78d6', edgecolor='white')
axes[1].set_title('Leaf Nodes vs Depth'); axes[1].set_xlabel('Max Depth'); axes[1].set_ylabel('Leaves'); axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('depth_analysis.png', dpi=150, bbox_inches='tight'); plt.show()
print("Saved: depth_analysis.png")

# 7b — Medicine distribution
fig, ax = plt.subplots(figsize=(12, 5))
mc = df['medicine'][mask].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(mc)))
bars = ax.barh(mc.index, mc.values, color=colors, edgecolor='white')
for b, v in zip(bars, mc.values):
    ax.text(b.get_width()+1, b.get_y()+b.get_height()/2, str(v), va='center', fontsize=10)
ax.set_title('Medicine Class Distribution'); ax.invert_yaxis()
plt.tight_layout(); plt.savefig('medicine_distribution.png', dpi=150, bbox_inches='tight'); plt.show()
print("Saved: medicine_distribution.png")

# 7c — Feature importances
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, clf, title in [(axes[0], clf_limited, f'Depth-limited n={best_n}'),
                        (axes[1], clf_unlimited, 'Unlimited')]:
    imp = pd.Series(clf.feature_importances_, index=ALL_SYMPTOMS)
    imp = imp[imp > 0].sort_values().tail(15)
    ax.barh(imp.index, imp.values, color='#2a78d6', edgecolor='white')
    ax.set_title(f'Top Features — {title}'); ax.set_xlabel('Gini importance'); ax.grid(alpha=0.3, axis='x')
plt.tight_layout(); plt.savefig('feature_importances.png', dpi=150, bbox_inches='tight'); plt.show()
print("Saved: feature_importances.png")

# 7d — Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, clf, title in [(axes[0], clf_limited, f'Depth-limited n={best_n}'),
                        (axes[1], clf_unlimited, 'Unlimited')]:
    cm = confusion_matrix(y_test, clf.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False, cmap='Blues', xticks_rotation=45)
    ax.set_title(title)
plt.tight_layout(); plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight'); plt.show()
print("Saved: confusion_matrices.png")

# 7e — Tree diagram (depth-limited only — unlimited is too large to render)
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(clf_limited, feature_names=list(ALL_SYMPTOMS), class_names=class_names,
          filled=True, rounded=True, fontsize=7, ax=ax, impurity=False)
ax.set_title(f'Decision Tree — Depth-limited (n={best_n})', fontsize=14)
plt.tight_layout(); plt.savefig('tree_limited.png', dpi=150, bbox_inches='tight'); plt.show()
print("Saved: tree_limited.png")

# =============================================================================
#  STEP 8 — Predict for a new patient
# =============================================================================

def predict_medicine(symptoms_list, model=clf_limited, label='Depth-limited'):
    row = np.zeros(len(ALL_SYMPTOMS), dtype=int)
    found, missing = [], []
    for s in symptoms_list:
        s = s.strip().lower().replace(' ', '_')
        if s in ALL_SYMPTOMS:
            row[ALL_SYMPTOMS.index(s)] = 1
            found.append(s)
        else:
            missing.append(s)

    # ── Walk the tree manually node by node ──────────────────────
    tree       = model.tree_
    feature    = tree.feature
    threshold  = tree.threshold
    children_l = tree.children_left
    children_r = tree.children_right

    node    = 0          # start at root
    path    = []
    step    = 1

    while children_l[node] != children_r[node]:   # not a leaf
        feat_idx  = feature[node]
        feat_name = ALL_SYMPTOMS[feat_idx]
        val       = row[feat_idx]
        direction = 'Yes' if val <= threshold[node] else 'No'
        # threshold for binary features is 0.5, so <= 0.5 means symptom absent
        present   = val > threshold[node]
        path.append(
            f"  Step {step}: Is '{feat_name}' present? "
            f"{'YES' if present else 'NO '} → go {'right' if present else 'left'}"
        )
        node  = children_r[node] if present else children_l[node]
        step += 1

    # At leaf
    leaf_counts = tree.value[node][0]
    total       = leaf_counts.sum()
    pred        = int(np.argmax(leaf_counts))
    proba_full  = model.predict_proba([row])[0]
    med         = le.inverse_transform([pred])[0]
    conf        = proba_full[pred]

    print(f"\n{'='*55}")
    print(f"  [{label}]")
    print(f"{'='*55}")
    print(f"  Symptoms entered : {found}")
    if missing:
        print(f"  Not recognised   : {missing}")
    print(f"\n  Decision path ({step-1} splits):")
    for p in path:
        print(p)
    print(f"\n  Leaf node reached:")
    print(f"    Training samples at this leaf : {int(total)}")
    print(f"    Class distribution            : ", end='')
    for i, cnt in enumerate(leaf_counts):
        if cnt > 0:
            print(f"{class_names[i]}={int(cnt)}", end='  ')
    print(f"\n\n  >>> Recommended medicine : {med}  ({conf:.0%} confidence) <<<")
    print()
    print("  Top 3 candidates:")
    for m, p in sorted(zip(class_names, proba_full), key=lambda x: -x[1])[:3]:
        bar = '█' * int(p * 20)
        print(f"    {bar:<20}  {p:.0%}  {m}")
    print(f"{'='*55}")
    return med

print("\n" + "=" * 60)
print("  STEP 8 — Sample predictions")
print("=" * 60)
predict_medicine(['fever', 'chills', 'vomiting', 'sweating', 'headache'])
predict_medicine(['fever', 'chills', 'vomiting', 'sweating', 'headache'], clf_unlimited, 'Unlimited')
predict_medicine(['skin_rash', 'itching', 'nodal_skin_eruptions'])
predict_medicine(['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'])
predict_medicine(['back_pain', 'constipation', 'abdominal_pain', 'yellow_urine'])

print("\nDone! All plots saved.")

# =============================================================================
#  STEP 9 — Interactive symptom input
# =============================================================================

def interactive_predict():
    print("\n" + "=" * 60)
    print("  INTERACTIVE MEDICINE PREDICTOR")
    print("=" * 60)
    print("Type your symptoms one by one and press Enter after each.")
    print("When done, just press Enter on a blank line.")
    print("Type 'list' to see all available symptoms.")
    print("Type 'quit' to exit.\n")

    while True:
        # Choose tree mode
        print("Select tree type:")
        print("  1 — Depth-limited (n={})".format(best_n))
        print("  2 — Unlimited")
        choice = input("Choice (1/2): ").strip()
        model = clf_unlimited if choice == '2' else clf_limited
        label = 'Unlimited' if choice == '2' else f'Depth-limited (n={best_n})'

        # Collect symptoms
        user_symptoms = []
        print(f"\nEnter symptoms (blank line when done):")
        while True:
            s = input(f"  Symptom {len(user_symptoms)+1}: ").strip().lower().replace(' ', '_')
            if s == 'quit':
                print("Exiting. Goodbye!")
                return
            if s == 'list':
                print("\nAvailable symptoms:")
                for i, sym in enumerate(sorted(ALL_SYMPTOMS), 1):
                    print(f"  {i:3}. {sym}")
                print()
                continue
            if s == '':
                break
            if s in ALL_SYMPTOMS:
                user_symptoms.append(s)
                print(f"       ✓ added '{s}'")
            else:
                # Fuzzy suggestion — find closest match
                close = [sym for sym in ALL_SYMPTOMS if s[:4] in sym or sym[:4] in s]
                print(f"       ✗ '{s}' not found.", end='')
                if close:
                    print(f" Did you mean: {', '.join(close[:3])}?")
                else:
                    print()

        if not user_symptoms:
            print("No valid symptoms entered. Try again.\n")
            continue

        # Run prediction
        predict_medicine(user_symptoms, model=model, label=label)

        # Ask to continue
        again = input("\nPredict for another patient? (y/n): ").strip().lower()
        if again != 'y':
            print("Exiting. Goodbye!")
            break
        print()

# Run the interactive predictor
interactive_predict()

# =============================================================================
#  STEP 10 — Visual decision path (matplotlib popup)
# =============================================================================

def visualize_decision_path(symptoms_list, model=clf_limited, label='Depth-limited'):
    """
    Draws the full decision tree and highlights the path taken for the given
    symptoms in red. Each visited node is coloured, and the final leaf is
    shown in green.
    """
    from sklearn.tree import _tree
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe

    # --- build row vector ---
    row = np.zeros(len(ALL_SYMPTOMS), dtype=int)
    found, missing = [], []
    for s in symptoms_list:
        s = s.strip().lower().replace(' ', '_')
        if s in ALL_SYMPTOMS:
            row[ALL_SYMPTOMS.index(s)] = 1
            found.append(s)
        else:
            missing.append(s)

    # --- walk the tree to collect visited node ids ---
    tree_      = model.tree_
    children_l = tree_.children_left
    children_r = tree_.children_right
    feature    = tree_.feature
    threshold  = tree_.threshold

    visited = set()
    visited_edges = []   # (parent, child, label)
    node = 0
    visited.add(node)
    while children_l[node] != children_r[node]:
        present = row[feature[node]] > threshold[node]
        next_node = children_r[node] if present else children_l[node]
        visited_edges.append((node, next_node, 'YES' if present else 'NO'))
        node = next_node
        visited.add(node)
    leaf_node = node

    # --- draw tree using plot_tree, then overlay highlights ---
    # Limit visual to depth-limited tree for readability;
    # for unlimited tree cap display depth at 6
    display_model = model
    max_display_depth = None if model.max_depth and model.max_depth <= 6 else 6

    fig, ax = plt.subplots(figsize=(26, 12))

    artists = plot_tree(
        display_model,
        feature_names=list(ALL_SYMPTOMS),
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
        impurity=False,
        max_depth=max_display_depth,
        node_ids=True,
    )

    # plot_tree returns one artist per node in tree order.
    # We highlight visited nodes by re-colouring their patch.
    n_nodes = tree_.node_count
    for i, artist in enumerate(artists):
        if i >= n_nodes:
            break
        if hasattr(artist, 'get_bbox_patch'):        # text with box
            bbox = artist.get_bbox_patch()
            if bbox is None:
                continue
            if i == leaf_node:
                bbox.set_facecolor('#4CAF50')        # green — final answer
                bbox.set_edgecolor('#1B5E20')
                bbox.set_linewidth(3)
                artist.set_color('white')
                artist.set_fontweight('bold')
            elif i in visited:
                bbox.set_facecolor('#FF7043')        # orange-red — visited
                bbox.set_edgecolor('#BF360C')
                bbox.set_linewidth(2.5)
                artist.set_color('white')

    # Title and legend
    ax.set_title(
        f'Decision Tree — {label}\n'
        f'Patient symptoms: {found}',
        fontsize=13, pad=14
    )
    legend_patches = [
        mpatches.Patch(facecolor='#FF7043', edgecolor='#BF360C', label='Path taken'),
        mpatches.Patch(facecolor='#4CAF50', edgecolor='#1B5E20', label='Predicted leaf'),
        mpatches.Patch(facecolor='#e8f4f8', edgecolor='#aaa',    label='Unvisited node'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    # Print result below title
    leaf_counts = tree_.value[leaf_node][0]
    pred_class  = class_names[int(np.argmax(leaf_counts))]
    conf        = model.predict_proba([row])[0][int(np.argmax(leaf_counts))]
    fig.text(0.5, 0.01,
             f'Recommended medicine: {pred_class}  ({conf:.0%} confidence)  |  '
             f'Leaf samples: {int(leaf_counts.sum())}',
             ha='center', fontsize=12, color='#1B5E20', fontweight='bold')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = f'decision_path_{label.replace(" ","_").replace("=","")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")
    if missing:
        print(f"(Symptoms not recognised and skipped: {missing})")
    return pred_class


# =============================================================================
#  STEP 11 — Interactive predictor with visualisation
# =============================================================================

def interactive_predict_visual():
    print("\n" + "=" * 60)
    print("  INTERACTIVE PREDICTOR  (with tree visualisation)")
    print("=" * 60)
    print("Commands:  'list' = show all symptoms | 'quit' = exit\n")

    while True:
        print("Select tree type:")
        print(f"  1 — Depth-limited (n={best_n})")
        print(  "  2 — Unlimited")
        choice = input("Choice (1/2): ").strip()
        model  = clf_unlimited if choice == '2' else clf_limited
        label  = 'Unlimited'  if choice == '2' else f'Depth-limited (n={best_n})'

        user_symptoms = []
        print(f"\nEnter symptoms (blank line when done):")
        while True:
            s = input(f"  Symptom {len(user_symptoms)+1}: ").strip().lower().replace(' ', '_')
            if s == 'quit':
                print("Bye!")
                return
            if s == 'list':
                print("\nAvailable symptoms:")
                for i, sym in enumerate(sorted(ALL_SYMPTOMS), 1):
                    print(f"  {i:3}. {sym}")
                print()
                continue
            if s == '':
                break
            if s in ALL_SYMPTOMS:
                user_symptoms.append(s)
                print(f"       ✓ '{s}' added")
            else:
                close = [sym for sym in ALL_SYMPTOMS if s[:4] in sym or sym[:4] in s]
                print(f"       ✗ '{s}' not found.", end='')
                if close:
                    print(f" Did you mean: {', '.join(close[:3])}?")
                else:
                    print()

        if not user_symptoms:
            print("No valid symptoms. Try again.\n")
            continue

        # Text output first
        predict_medicine(user_symptoms, model=model, label=label)

        # Then visual
        print("\nOpening tree visualisation...")
        visualize_decision_path(user_symptoms, model=model, label=label)

        again = input("\nAnother patient? (y/n): ").strip().lower()
        if again != 'y':
            print("Bye!")
            break
        print()


interactive_predict_visual()