import argparse
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import cv2
from screeninfo import get_monitors
# Outils pour t-SNE et visualisation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


from sklearn.metrics import f1_score
import os
import json
import cv2
import numpy as np
from PIL import Image
import datetime
import pandas as pd








# -------------------------------------------------------------------
# 4.1) FONCTION POUR COMPTER LES PARAMÈTRES DU MODÈLE
# -------------------------------------------------------------------
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trunk_params = sum(p.numel() for p in model.trunk.parameters() if p.requires_grad)
    print("==== Comptage des paramètres ====")
    print(f"Paramètres totaux du modèle : {total_params}")
    print(f"Paramètres du tronc : {trunk_params}")
    print("Paramètres par tête de tâche :")
    for task, head in model.task_heads.items():
        head_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        # On récupère in_channels et out_channels pour calcul théorique
        in_channels = head.attention.conv_mask.weight.shape[1]
        out_channels = head.final_conv.weight.shape[0]
        # Pour SpatialAttention : (in_channels + 1) paramètres
        # Pour final_conv : (out_channels * in_channels * 4 * 4) + out_channels paramètres
        theoretical = (in_channels + 1) + (out_channels * in_channels * 16 + out_channels)
        print(f"  - Tâche '{task}' : {head_params} paramètres (théoriques: {theoretical})")
    print("=================================")





# -------------------------------------------------------------------
# 6) CHARGEMENT DU MODELE
# -------------------------------------------------------------------
def load_model_weights(model, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Fichier introuvable : {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Charger les poids en mode non strict pour ne prendre que ceux correspondant aux têtes existantes
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Modèle chargé depuis {checkpoint_path}")


# -------------------------------------------------------------------
# 7) CALCUL EMBEDDINGS (pour t-SNE, clustering, etc.)
# -------------------------------------------------------------------
def compute_embeddings_with_paths(model, loader, device, tasks_json, per_task_tsne=False):
    model.eval()
    if per_task_tsne:
        embeddings_dict = {tname: [] for tname in tasks_json.keys()}
        labels_dict = {tname: [] for tname in tasks_json.keys()}
        img_paths_dict = {tname: [] for tname in tasks_json.keys()}
    else:
        all_embeddings = []
        all_labels = []
        img_paths = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            outputs, task_emb = model(inputs, return_task_embeddings=True)
            batch_size = inputs.size(0)
            if isinstance(loader.dataset, Subset):
                indices = loader.dataset.indices[batch_idx * loader.batch_size : batch_idx * loader.batch_size + batch_size]
                batch_img_paths = [loader.dataset.dataset.samples[idx][0] for idx in indices]
            else:
                batch_img_paths = [loader.dataset.samples[idx][0] for idx in range(batch_idx * loader.batch_size,
                                        batch_idx * loader.batch_size + batch_size)]
            if per_task_tsne:
                for tname in tasks_json.keys():
                    emb_batch = task_emb[tname].cpu().numpy()
                    label_batch = labels[tname].clone()
                    label_batch[label_batch < 0] = -1
                    for i in range(batch_size):
                        embeddings_dict[tname].append(emb_batch[i])
                        labels_dict[tname].append(int(label_batch[i].item()) if label_batch[i].item() >= 0 else -1)
                        img_paths_dict[tname].append(batch_img_paths[i])
            else:
                first_task = list(tasks_json.keys())[0]
                emb_batch = task_emb[first_task].cpu().numpy()
                label_batch = labels[first_task].clone()
                label_batch[label_batch < 0] = -1
                all_embeddings.append(emb_batch)
                all_labels.append(label_batch.cpu().numpy())
                img_paths.extend(batch_img_paths)
    if per_task_tsne:
        for tname in embeddings_dict.keys():
            embeddings_dict[tname] = np.array(embeddings_dict[tname])
            labels_dict[tname] = np.array(labels_dict[tname])
        return embeddings_dict, labels_dict, img_paths_dict
    else:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_embeddings, all_labels, img_paths

# -------------------------------------------------------------------
# 8) FONCTION TSNE SIMPLE (non-interactif)
# -------------------------------------------------------------------


def perform_tsne(attentive_embeddings, labels, tasks, colors=None, results_dir='results', task_name=None):
    """
    Réalise un t-SNE sur les attentive embeddings (issus du mécanisme d'attention) et trace le scatter plot.

    Args:
        attentive_embeddings (np.array): Tableau des embeddings de la partie attention, de forme (N, C, H, W).
        labels (np.array): Tableau des labels associés (N,).
        tasks (dict): Dictionnaire associant chaque tâche à la liste de ses classes.
        colors (list, optional): Liste de couleurs à utiliser pour le plot.
        results_dir (str): Répertoire où sauvegarder le graphique.
        task_name (str, optional): Nom de la tâche pour laquelle tracer le t-SNE.
    """
    print("Réalisation de t-SNE sur les attentive embeddings...")
    # Aplatir les attentive embeddings issus du module d'attention
    embeddings_flat = attentive_embeddings.reshape(attentive_embeddings.shape[0], -1)
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings_flat)

    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Définir la palette de couleurs
    if colors and len(colors) >= num_classes:
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    else:
        color_map = {label: plt.cm.tab20(i / num_classes) for i, label in enumerate(unique_labels)}

    # Récupérer la liste des classes (on prend celle de la première tâche si non précisée)
    if task_name:
        class_names = tasks[task_name]
    else:
        class_names = tasks[list(tasks.keys())[0]]

    # Tracer les points par classe
    for label in unique_labels:
        inds = labels == label
        plt.scatter(embeddings_2d[inds, 0], embeddings_2d[inds, 1],
                    label=class_names[label], color=color_map[label])
    plt.legend()

    # Sauvegarder le plot
    if task_name:
        tsne_plot_path = os.path.join(results_dir, f'tsne_plot_{task_name.replace(" ", "_")}.png')
    else:
        tsne_plot_path = os.path.join(results_dir, 'tsne_plot.png')
    plt.savefig(tsne_plot_path)
    plt.show()
    print(f"t-SNE plot saved to '{tsne_plot_path}'")


def plot_tsne_interactive(attentive_embeddings_data, labels_data, tasks, img_paths_data, colors=None, num_clusters=None,
                          save_dir='results'):
    """
    Ouvre une interface interactive Tkinter pour explorer un t-SNE calculé sur les attentive embeddings.

    L'interface permet :
      - de choisir une tâche (si plusieurs sont présentes),
      - de recalculer le t-SNE pour la tâche sélectionnée,
      - de zoomer/dézoomer,
      - de tracer un polygone sur le plot pour sélectionner des points,
      - d'afficher la ou les images associées à chaque point sélectionné.

    Args:
        attentive_embeddings_data (dict): Dictionnaire associant chaque tâche à ses attentive embeddings (numpy array de forme (N, C, H, W)).
        labels_data (dict): Dictionnaire associant chaque tâche à ses labels (numpy array).
        tasks (dict): Dictionnaire associant chaque tâche à la liste de ses classes.
        img_paths_data (dict): Dictionnaire associant chaque tâche à la liste des chemins d'images.
        colors (list, optional): Liste de couleurs à utiliser pour le plot.
        num_clusters (int, optional): (Non utilisé ici, mais peut être étendu pour le clustering interactif).
        save_dir (str): Répertoire où sauvegarder d'éventuelles sorties (ex. fichiers JSON des points sélectionnés).
    """
    import matplotlib
    matplotlib.use('TkAgg')
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import ttk, colorchooser
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.path import Path
    from matplotlib.widgets import PolygonSelector
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import os, json, numpy as np

    # Déterminer si l'on travaille avec un dictionnaire (plusieurs tâches) ou un seul tableau (tâche unique)
    if isinstance(attentive_embeddings_data, dict):
        single_task_mode = (len(attentive_embeddings_data) == 1)
        if single_task_mode:
            current_task_name = list(attentive_embeddings_data.keys())[0]
        else:
            current_task_name = None
    else:
        single_task_mode = True
        current_task_name = None

    tsne_results = None
    labels = None
    class_names = None
    unique_labels = None
    scatter = None
    color_map = None
    img_paths = None
    filename_to_path = None
    polygon = []
    polygon_selector = None
    polygon_cleared = True

    # Création des frames pour l'interface
    root = tk.Tk()
    root.title("Interactive t-SNE with Images")
    left_frame = tk.Frame(root)
    left_frame.grid(row=0, column=0, sticky='nsew')
    right_frame = tk.Frame(root)
    right_frame.grid(row=0, column=1, sticky='nsew')

    # Intégration de la figure matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Zone d'affichage d'image et informations
    img_label = tk.Label(right_frame)
    img_label.pack(pady=10)
    label_text = tk.StringVar()
    label_label = tk.Label(right_frame, textvariable=label_text, justify='left')
    label_label.pack()
    inside_points_label = tk.StringVar()
    inside_points_count_label = tk.Label(right_frame, textvariable=inside_points_label)
    inside_points_count_label.pack()

    dropdown_points = []
    dropdown = ttk.Combobox(right_frame, state="readonly")
    dropdown.pack(fill='x', pady=5)
    dropdown.bind("<<ComboboxSelected>>", lambda event: on_dropdown_select())

    def change_class_color():
        selected = class_selector.get()
        if selected:
            label_str = selected.split(':')[0]
            label_val = int(label_str)
            color_code = colorchooser.askcolor(title="Choisir une couleur")[1]
            if color_code:
                color_map[label_val] = color_code
                scatter.set_color([color_map[int(lbl)] for lbl in labels])
                ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=class_names[int(lbl)],
                                              markerfacecolor=color_map[int(lbl)], markersize=10) for lbl in
                                   unique_labels])
                canvas.draw()

    class_selector_label = tk.Label(right_frame, text="Sélectionnez une classe :")
    class_selector_label.pack(pady=5)
    class_selector = ttk.Combobox(right_frame, state="readonly")
    class_selector.pack(pady=5)
    change_color_button = tk.Button(right_frame, text="Changer la couleur de la classe", command=change_class_color)
    change_color_button.pack(pady=5)

    button_frame = tk.Frame(right_frame)
    button_frame.pack(pady=10)
    close_button = tk.Button(button_frame, text="Fermer le polygone", command=lambda: analyze_polygon())
    close_button.pack(side='left', padx=5)
    clear_button = tk.Button(button_frame, text="Effacer le polygone", command=lambda: clear_polygon())
    clear_button.pack(side='left', padx=5)

    def clear_polygon():
        nonlocal polygon_selector, polygon_cleared
        polygon.clear()
        if polygon_selector:
            polygon_selector.disconnect_events()
            polygon_selector.set_visible(False)
            del polygon_selector
            polygon_selector = None
        while ax.patches:
            ax.patches.pop().remove()
        fig.canvas.draw()
        inside_points_label.set("")
        label_text.set("")
        img_label.config(image='')
        dropdown.set('')
        dropdown['values'] = []
        polygon_cleared = True

    def update_plot(task_name):
        nonlocal tsne_results, labels, class_names, unique_labels, scatter, color_map, img_paths, filename_to_path, current_task_name
        current_task_name = task_name
        ax.clear()
        # Utiliser les attentive embeddings pour la tâche sélectionnée
        if isinstance(attentive_embeddings_data, dict):
            embeddings = attentive_embeddings_data[task_name]
            labels_local = labels_data[task_name]
            img_paths = img_paths_data[task_name]
            class_names = tasks[task_name]
        else:
            embeddings = attentive_embeddings_data
            labels_local = labels_data
            img_paths = img_paths_data
            class_names = tasks[list(tasks.keys())[0]]
        filename_to_path = {os.path.basename(path): path for path in img_paths}
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
        tsne_results = tsne.fit_transform(embeddings_flat)
        labels = labels_local
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        if colors and len(colors) >= num_classes:
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        else:
            color_palette = plt.cm.get_cmap("tab20", num_classes)
            color_map = {label: color_palette(i / num_classes) for i, label in enumerate(unique_labels)}
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[color_map[int(label)] for label in labels],
                             picker=True)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[int(label)],
                                      markerfacecolor=color_map[int(label)], markersize=10) for label in unique_labels]
        ax.legend(handles=legend_elements)
        ax.set_title(f"t-SNE pour la tâche : {task_name}" if task_name else "t-SNE")
        canvas.draw()
        class_selector['values'] = [f"{label}: {class_names[label]}" for label in unique_labels]
        if unique_labels.size > 0:
            class_selector.current(0)
        clear_polygon()

    def on_task_select(event):
        selected_task = task_selector.get()
        update_plot(selected_task)

    def onpick(event):
        ind = event.ind[0]
        img_path = img_paths[ind]
        display_image(img_path, class_names[int(labels[ind])])

    fig.canvas.mpl_connect('pick_event', onpick)

    def enable_polygon_selector(event):
        nonlocal polygon_selector, polygon_cleared
        if event.button == 3:  # clic droit
            if polygon_selector is None or polygon_cleared:
                polygon_selector = PolygonSelector(ax, onselect=onselect, useblit=True)
                polygon_cleared = False
                print("Sélecteur de polygone activé.")

    def onselect(verts):
        polygon.clear()
        polygon.extend(verts)
        print("Sommets du polygone:", verts)

    def analyze_polygon():
        if len(polygon) < 3:
            print("Polygone non fermé. Sélectionnez au moins 3 points.")
            return
        inside_points = []
        outside_points = []
        polygon_path = Path(polygon)
        for i, (x, y) in enumerate(tsne_results):
            point = (x, y)
            if polygon_path.contains_point(point):
                inside_points.append({"path": img_paths[i], "class": class_names[int(labels[i])], "position": point})
            else:
                outside_points.append({"path": img_paths[i], "class": class_names[int(labels[i])], "position": point})
        for point in inside_points:
            point['filename'] = os.path.basename(point['path'])
            del point['path']
        for point in outside_points:
            point['filename'] = os.path.basename(point['path'])
            del point['path']
        filename_suffix = current_task_name.replace(' ', '_') if current_task_name else 'task'
        with open(os.path.join(save_dir, f"inside_polygon_{filename_suffix}.json"), "w") as f:
            json.dump(inside_points, f)
        with open(os.path.join(save_dir, f"outside_polygon_{filename_suffix}.json"), "w") as f:
            json.dump(outside_points, f)
        inside_points_label.set(f"Points à l'intérieur du polygone: {len(inside_points)}")
        update_dropdown(inside_points)

    def update_dropdown(inside_points):
        dropdown_values = [f"{point['filename']} ({point['class']})" for point in inside_points]
        dropdown['values'] = dropdown_values
        dropdown_points.clear()
        dropdown_points.extend(inside_points)
        if dropdown_values:
            dropdown.current(0)
            on_dropdown_select()

    def on_dropdown_select():
        selection = dropdown.current()
        if selection >= 0:
            point = dropdown_points[selection]
            img_path = filename_to_path[point['filename']]
            display_image(img_path, point['class'])

    def display_image(img_path, label):
        img = Image.open(img_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
        label_text.set(f"Label: {label}\nFichier: {os.path.basename(img_path)}")

    fig.canvas.mpl_connect('button_press_event', enable_polygon_selector)

    def on_key_press(event):
        if event.key == '+':
            zoom(1.2)
        elif event.key == '-':
            zoom(0.8)

    def on_scroll(event):
        if event.button == 'up':
            zoom(1.1)
        elif event.button == 'down':
            zoom(0.9)

    def zoom(factor):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = np.mean(xlim)
        ydata = np.mean(ylim)
        width = (xlim[1] - xlim[0]) * factor
        height = (ylim[1] - ylim[0]) * factor
        ax.set_xlim([xdata - width / 2, xdata + width / 2])
        ax.set_ylim([ydata - height / 2, ydata + height / 2])
        canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    root.grid_columnconfigure(0, weight=3)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)
    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)

    if not single_task_mode:
        task_selector_label = tk.Label(right_frame, text="Sélectionnez une tâche :")
        task_selector_label.pack(pady=5)
        task_selector = ttk.Combobox(right_frame, state="readonly", values=list(tasks.keys()))
        task_selector.pack(pady=5)
        task_selector.bind("<<ComboboxSelected>>", on_task_select)

    if single_task_mode:
        update_plot(list(tasks.keys())[0])
    else:
        initial_task = list(tasks.keys())[0]
        task_selector.set(initial_task)
        update_plot(initial_task)

    root.mainloop()


# -------------------------------------------------------------------
# 5) FONCTION DE TEST AVEC OPTIONS Grad-CAM ET Integrated Gradients
# -------------------------------------------------------------------
def test_classifier(model, test_loader, criterions, writer, save_dir, device, tasks_json,
                    prob_threshold=0.5, visualize_gradcam=False, save_gradcam_images=False,
                    measure_time=False, save_test_images=False, gradcam_task=None, colormap='hot',
                    integrated_gradients=False, integrated_gradients_task=None):

    model.eval()
    total_loss = 0.0
    total_samples = 0
    times = []

    all_preds = {t: [] for t in tasks_json.keys()}
    all_labels = {t: [] for t in tasks_json.keys()}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Préparation de Grad-CAM si activé
    if visualize_gradcam:
        if gradcam_task is None:
            gradcam_task = list(tasks_json.keys())[0]
        if gradcam_task not in tasks_json:
            raise ValueError(f"La tâche '{gradcam_task}' n'existe pas.")
        gradcam_model = TaskSpecificModel(model, gradcam_task).to(device)
        gradcam_model.eval()
        target_layer = None
        for layer in reversed(list(gradcam_model.model.trunk)):
            if isinstance(layer, nn.Conv2d):
                target_layer = layer
                break
        if target_layer is None:
            raise ValueError("No Conv2d layer found for Grad-CAM.")
        grad_cam = GradCAM(model=gradcam_model, target_layers=[target_layer])

    # Préparation d'Integrated Gradients
    if integrated_gradients:
        from captum.attr import IntegratedGradients
        ig_models = {}
        ig = {}

        # Tâches concernées par IG
        if integrated_gradients_task is not None:
            tasks_to_compute = [integrated_gradients_task]
        else:
            tasks_to_compute = list(tasks_json.keys())

        for t in tasks_to_compute:
            ig_models[t] = TaskSpecificModel(model, t).to(device)
            ig_models[t].eval()
            ig[t] = IntegratedGradients(ig_models[t])

    for batch_idx, (inputs, labels) in enumerate(test_loader):
        start_time = time.time()

        inputs = inputs.to(device)
        inputs.requires_grad = True

        with torch.no_grad():
            outputs = model(inputs)

        loss = 0.0
        batch_size = inputs.size(0)
        preds_dict = {}
        max_probs_dict = {}
        labels_dict = {}

        # Calcul de la prédiction / CrossEntropy
        for t, criterion in criterions.items():
            t_labels = labels[t]
            if t_labels is not None:
                t_labels = t_labels.to(device)
                t_out = outputs[t]
                t_loss = criterion(t_out, t_labels)
                loss += t_loss

                probabilities = torch.softmax(t_out, dim=1)
                max_probs, preds = torch.max(probabilities, dim=1)

                unknown_mask = max_probs < prob_threshold
                preds[unknown_mask] = -1

                all_preds[t].extend(preds.cpu().numpy())
                all_labels[t].extend(t_labels.cpu().numpy())

                preds_dict[t] = preds.cpu().numpy()
                max_probs_dict[t] = max_probs.cpu().numpy()
                labels_dict[t] = t_labels.cpu().numpy()
            else:
                preds_dict[t] = np.array([-1]*batch_size)
                max_probs_dict[t] = np.array([-1]*batch_size)
                labels_dict[t] = np.array([-1]*batch_size)

        end_time = time.time()
        times.append(end_time - start_time)

        if integrated_gradients:
            for i in range(batch_size):
                # 1) Lire l'image avec PIL en RGB
                if isinstance(test_loader.dataset, Subset):
                    img_path = test_loader.dataset.dataset.samples[
                        test_loader.dataset.indices[batch_idx * test_loader.batch_size + i]
                    ][0]
                else:
                    img_path = test_loader.dataset.samples[batch_idx * test_loader.batch_size + i][0]

                # On récupère l'image en numpy [H,W,3] en RGB
                orig_img_rgb = np.array(Image.open(img_path).convert('RGB'))

                # 2) Convertir cette image en BGR, car OpenCV utilise BGR
                orig_img_bgr = cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2BGR)

                # Sélection des tâches concernées par Integrated Gradients
                if integrated_gradients_task is not None:
                    tasks_ig = [integrated_gradients_task]
                else:
                    tasks_ig = list(tasks_json.keys())

                for t in tasks_ig:
                    input_tensor = inputs[i].unsqueeze(0)
                    baseline = torch.zeros_like(input_tensor)
                    target = int(labels_dict[t][i]) if labels_dict[t][i] >= 0 else 0

                    # Calcul des attributions IG
                    attributions = ig[t].attribute(input_tensor, baseline, target=target)
                    attr_np = attributions.squeeze().cpu().detach().numpy()

                    # Moyenne sur les canaux si >2D
                    if attr_np.ndim > 2:
                        attr_np = np.mean(attr_np, axis=0)

                    # Normaliser [0..1]
                    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)

                    # Générer la heatmap (BGR direct depuis cv2.applyColorMap)
                    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * attr_np), cv2.COLORMAP_JET)

                    # Optionnel : si vous voulez tout faire en BGR, on ne convertit pas en RGB
                    # => On reste cohérent pour addWeighted
                    # heatmap_bgr est donc [H,W,3] BGR

                    # Redimensionner la heatmap pour correspondre à la taille d'origine
                    h, w = orig_img_bgr.shape[:2]  # (H,W,3)
                    heatmap_bgr = cv2.resize(heatmap_bgr, (w, h))

                    # 3) Faire le blending (addWeighted) en BGR
                    #   => alpha=0.2 => 80% image d'origine, 20% heatmap
                    overlay_bgr = cv2.addWeighted(orig_img_bgr, 0.8, heatmap_bgr, 0.2, 0)

                    # 4) Sauvegarder en BGR (cv2.imwrite attend BGR) => couleurs cohérentes
                    true_label = "Unknown" if labels_dict[t][i] == -1 else tasks_json[t][labels_dict[t][i]]
                    ig_folder = os.path.join(save_dir, "IntegratedGradients", t, true_label)
                    if not os.path.exists(ig_folder):
                        os.makedirs(ig_folder)

                    ig_save_path = os.path.join(
                        ig_folder,
                        f"IntegratedGrad_{t}_{batch_idx * test_loader.batch_size + i}.jpg"
                    )
                    cv2.imwrite(ig_save_path, overlay_bgr)

        # (2) Sauvegarde des images annotées et Grad-CAM
        for i in range(batch_size):
            idx = batch_idx * test_loader.batch_size + i
            if isinstance(test_loader.dataset, Subset):
                img_path = test_loader.dataset.dataset.samples[
                    test_loader.dataset.indices[idx]
                ][0]
            else:
                img_path = test_loader.dataset.samples[idx][0]

            img = Image.open(img_path)
            img_np = np.array(img.convert('RGB'))

            if save_test_images or (visualize_gradcam and save_gradcam_images):
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Sauvegarde d'une version annotée
            if save_test_images:
                annotated_img = img_cv.copy()
                y_start = 30
                y_step = 30
                for j, (t, clist) in enumerate(tasks_json.items()):
                    label_idx = labels_dict[t][i]
                    pred_idx = preds_dict[t][i]
                    prob = max_probs_dict[t][i]

                    true_label = "Unknown" if label_idx == -1 else clist[label_idx]
                    pred_label = "Unknown" if pred_idx == -1 else clist[pred_idx]
                    text = f"{t} - True: {true_label}, Pred: {pred_label}, Prob: {prob:.2f}"

                    y_pos = y_start + j * y_step
                    cv2.putText(
                        annotated_img,
                        text,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

                main_label = "Unknown"
                for t in tasks_json.keys():
                    if t.lower() == "weather type":
                        if labels_dict[t][i] == -1:
                            main_label = "Unknown"
                        else:
                            main_label = tasks_json[t][labels_dict[t][i]]
                        break

                save_folder = os.path.join(save_dir, main_label)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                out_img_path = os.path.join(save_folder, f"test_image_{idx}.jpg")
                cv2.imwrite(out_img_path, annotated_img)

            # Grad-CAM
            if visualize_gradcam and save_gradcam_images:
                input_tensor = inputs[i].unsqueeze(0)
                t_idx = labels_dict[gradcam_task][i]
                pred_val = preds_dict[gradcam_task][i]
                prob = max_probs_dict[gradcam_task][i]

                true_label = "Unknown" if t_idx == -1 else tasks_json[gradcam_task][t_idx]
                pred_label = "Unknown" if pred_val == -1 else tasks_json[gradcam_task][pred_val]

                text = f"{gradcam_task} - True: {true_label}, Pred: {pred_label}, Prob: {prob:.2f}"

                target = [ClassifierOutputTarget(t_idx)]
                grayscale_cam = grad_cam(input_tensor=input_tensor, targets=target)[0]

                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
                h, w = img_np.shape[:2]
                cam_resized = cv2.resize(grayscale_cam, (w, h))

                if colormap not in colormap_dict:
                    cmap_code = cv2.COLORMAP_HOT
                else:
                    cmap_code = colormap_dict[colormap]

                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cmap_code)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                visualization = 0.5 * heatmap + 0.5 * img_np
                visualization = np.clip(visualization, 0, 255).astype(np.uint8)

                cv2.putText(visualization, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                gradcam_folder = os.path.join(save_dir, "GradCAM", true_label)
                if not os.path.exists(gradcam_folder):
                    os.makedirs(gradcam_folder)

                gradcam_save_path = os.path.join(gradcam_folder, f"GradCAM_{idx}.jpg")
                cv2.imwrite(gradcam_save_path, visualization)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples

    metrics = {}
    for t in tasks_json.keys():
        if len(all_preds[t]) > 0:
            preds_np = np.array(all_preds[t])
            labels_np = np.array(all_labels[t])
            valid = preds_np != -1
            if valid.sum() > 0:
                acc = np.mean(preds_np[valid] == labels_np[valid])
                prec = precision_score(labels_np[valid], preds_np[valid], average='weighted', zero_division=0)
                rec = recall_score(labels_np[valid], preds_np[valid], average='weighted', zero_division=0)
                f1 = f1_score(labels_np[valid], preds_np[valid], average='weighted', zero_division=0)
                conf = confusion_matrix(labels_np[valid], preds_np[valid], labels=list(range(len(tasks_json[t]))))
            else:
                acc = prec = rec = f1 = 0.0
                conf = np.zeros((len(tasks_json[t]), len(tasks_json[t])))
            metrics[t] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": conf.tolist()
            }
            print(f"Tâche {t} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
            print(f"Matrice de confusion pour {t}:\n{conf}\n")
        else:
            metrics[t] = {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "confusion_matrix": None
            }

    valid_accs = [m["accuracy"] for m in metrics.values() if m["accuracy"] is not None]
    avg_accuracy = float(np.mean(valid_accs)) if valid_accs else 0.0

    print(f"Performance moyenne - Acc: {avg_accuracy:.4f}")
    metrics["average"] = {"accuracy": avg_accuracy}
    metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métriques enregistrées dans {metrics_path}")

    if writer:
        writer.add_scalar("Test/Loss", average_loss)
        writer.add_scalar("Test/Average_Accuracy", avg_accuracy)
        for t, m in metrics.items():
            if m["accuracy"] is not None and t != "average":
                writer.add_scalar(f"Test/{t}_Accuracy", m["accuracy"])
                writer.add_scalar(f"Test/{t}_Precision", m["precision"])
                writer.add_scalar(f"Test/{t}_Recall", m["recall"])
                writer.add_scalar(f"Test/{t}_F1_Score", m["f1_score"])

    if measure_time:
        times_path = os.path.join(save_dir, "times_test.json")
        with open(times_path, 'w') as f:
            json.dump(times, f, indent=4)
        print(f"Temps moyen par batch: {np.mean(times):.4f}s, total: {np.sum(times):.4f}s")



# -------------------------------------------------------------------
# 11) CAMERA
# -------------------------------------------------------------------
def run_camera(model, tasks_json, save_dir, prob_threshold, measure_time, camera_index,
               kalman_filter=False, save_camera_video=False):
    device = model.device if hasattr(model, 'device') else torch.device('cpu')
    model.eval()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la caméra {camera_index}")
        return

    # Récupération des dimensions de l'écran (si nécessaire)
    from screeninfo import get_monitors
    monitors = get_monitors()
    screen = monitors[0]
    screen_width = screen.width
    screen_height = screen.height

    # Créer la fenêtre de la caméra en mode normal pour que les contrôles soient visibles
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    # Variable pour suivre l'état plein écran
    full_screen_state = False

    # Variables de contrôle pour l'enregistrement
    recording = False
    video_writer = None

    # Création de l'interface de contrôle avec Tkinter
    control_window = tk.Tk()
    control_window.title("Contrôle Enregistrement")

    # Variable Tkinter pour le statut d'enregistrement
    rec_var = tk.BooleanVar(value=False)
    # Zone de texte pour le nom de la vidéo
    video_name_var = tk.StringVar()

    def toggle_recording():
        nonlocal recording, video_writer
        recording = not recording
        rec_var.set(recording)
        if recording:
            btn_toggle.config(text="Arrêter l'enregistrement")
        else:
            btn_toggle.config(text="Démarrer l'enregistrement")
            if video_writer is not None:
                video_writer.release()
                video_writer = None

    def toggle_fullscreen():
        nonlocal full_screen_state
        if not full_screen_state:
            cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            btn_fullscreen.config(text="Quitter le plein écran")
        else:
            cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            btn_fullscreen.config(text="Plein écran")
        full_screen_state = not full_screen_state

    # Interface de contrôle
    lbl = ttk.Label(control_window, text="Nom de la vidéo (optionnel) :")
    lbl.pack(padx=10, pady=5)
    entry_video = ttk.Entry(control_window, textvariable=video_name_var, width=30)
    entry_video.pack(padx=10, pady=5)

    btn_toggle = ttk.Button(control_window, text="Démarrer l'enregistrement", command=toggle_recording)
    btn_toggle.pack(padx=10, pady=5)

    btn_fullscreen = ttk.Button(control_window, text="Plein écran", command=toggle_fullscreen)
    btn_fullscreen.pack(padx=10, pady=5)

    # Positionnement de la fenêtre de contrôle (peut être ajusté)
    control_window.geometry("300x200+50+50")

    times = []

    if save_camera_video:
        print("Option --save_video activée. Utilisez le bouton d'enregistrement pour démarrer/arrêter la vidéo.")
    else:
        print("Enregistrement vidéo désactivé (--save_video non spécifié).")

    # Préparation du Kalman Filter si activé
    if kalman_filter:
        from pykalman import KalmanFilter
        kf_dict = {}
        state_means = {}
        state_cov = {}
        for tn, clist in tasks_json.items():
            nb_cls = len(clist)
            kf_dict[tn] = KalmanFilter(initial_state_mean=np.zeros(nb_cls),
                                       initial_state_covariance=np.eye(nb_cls),
                                       n_dim_obs=nb_cls)
            state_means[tn] = np.zeros(nb_cls)
            state_cov[tn] = np.eye(nb_cls)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_t = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        img_tens = transform(pil_img).unsqueeze(0).to(device)
        outputs = model(img_tens)
        text_lines = []
        for tn, output in outputs.items():
            probas = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
            if kalman_filter:
                state_means[tn], state_cov[tn] = kf_dict[tn].filter_update(
                    state_mean=state_means[tn],
                    state_covariance=state_cov[tn],
                    observation=probas
                )
                probas = state_means[tn]
            pred_idx = np.argmax(probas)
            pred_prob = probas[pred_idx]
            pred_label = "Unknown" if pred_prob < prob_threshold else tasks_json[tn][pred_idx]
            text_lines.append(f"{tn}: {pred_label} ({pred_prob:.2f})")
        end_t = time.time()
        times.append(end_t - start_t)

        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        y0 = 30
        y_step = 40
        for i, line in enumerate(text_lines):
            y_pos = y0 + i * y_step
            cv2.putText(frame_resized, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if save_camera_video:
            if recording and video_writer is None:
                # Si l'utilisateur n'a pas saisi de nom, on génère un nom avec timestamp
                vname = video_name_var.get().strip()
                if vname == "":
                    vname = f"video_{int(time.time())}"
                video_path = os.path.join(save_dir, f"{vname}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (screen_width, screen_height))
                print(f"Enregistrement démarré: {video_path}")
            elif not recording and video_writer is not None:
                video_writer.release()
                print("Enregistrement arrêté.")
                video_writer = None

            if video_writer is not None:
                video_writer.write(frame_resized)

        cv2.imshow("Camera", frame_resized)
        # Mise à jour de l'interface de contrôle Tkinter
        control_window.update()
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print("Enregistrement vidéo terminé.")
    cv2.destroyAllWindows()

    if measure_time and len(times) > 0:
        out_times = os.path.join(save_dir, "times_camera.json")
        with open(out_times, 'w') as f:
            import json
            json.dump(times, f, indent=4)
        print(f"Temps moyen: {np.mean(times):.4f}s, total: {np.sum(times):.4f}s")

    # Fermeture de l'interface de contrôle
    control_window.destroy()




def map_folder_to_class(folder_name, class_list):
    """
    Essaie de faire correspondre le nom du dossier (ground truth)
    à l'une des classes en vérifiant si le nom du dossier est contenu
    dans le nom de la classe (sans tenir compte de la casse).
    """
    folder_lower = folder_name.lower()
    for cls in class_list:
        if folder_lower in cls.lower():
            return cls
    return None


def test_folder_predictions(model, tasks, test_folder, transform, device, save_dir,
                            save_test_images=False, target_task=None):
    """
    Parcourt récursivement le dossier test_folder et effectue les prédictions.

    - Si target_task est précisé, on traite uniquement cette tâche pour :
         • l'annotation (images sauvegardées dans un sous-dossier portant le nom de la classe prédite),
         • le calcul des scores F1 (par classe et global) basé sur la ground truth extraite de la structure du dossier.
    - Sinon, le modèle effectue des prédictions pour toutes les tâches.
         • Les images sont rangées selon la tâche par défaut (la première tâche),
         • L'annotation affiche les prédictions pour toutes les tâches,
         • Le JSON final "folder_predictions.json" contient, pour chaque tâche, le nombre d'images par classe et
           les scores F1 (global et par classe) basés sur la ground truth extraite.
         • De plus, un second fichier "all_predictions.json" est généré, donnant pour chaque image
           (identifiée par son chemin relatif par rapport au dossier de test) l'ensemble des prédictions.

    Args:
        model (torch.nn.Module): Modèle multi-tâches chargé.
        tasks (dict): Dictionnaire associant chaque tâche à la liste de ses classes.
        test_folder (str): Chemin vers le dossier contenant les images de test.
        transform: Transformation à appliquer aux images.
        device: Appareil utilisé (CPU ou GPU).
        save_dir (str): Répertoire de sauvegarde des résultats.
        save_test_images (bool): Si True, enregistre les images annotées.
        target_task (str, optional): Nom de la tâche sur laquelle réaliser le test.
    """
    # Définir la tâche utilisée pour le classement et l'évaluation
    if target_task is not None:
        tasks_to_evaluate = {target_task: tasks[target_task]}
        folder_task = target_task
    else:
        tasks_to_evaluate = tasks  # toutes les tâches
        folder_task = list(tasks.keys())[0]  # on organise les images par la première tâche

    # Initialisation des dictionnaires pour chaque tâche évaluée
    predictions_by_task = {t: {} for t in tasks_to_evaluate.keys()}  # comptage par classe
    gt_by_task = {t: [] for t in tasks_to_evaluate.keys()}  # ground truth extraites
    pred_gt_by_task = {t: [] for t in tasks_to_evaluate.keys()}  # prédictions associées aux GT

    results = {}  # résultats complets par image (pour annotation)

    # Dossier de sauvegarde des images annotées
    if save_test_images:
        annotated_base_dir = os.path.join(save_dir, "annotated_images")
        os.makedirs(annotated_base_dir, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    # Parcours récursif des fichiers dans test_folder
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue
            img_path = os.path.join(root, file)
            # On calcule le chemin relatif pour identifier l'image dans all_predictions.json
            rel_path = os.path.relpath(img_path, test_folder)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Erreur lors du chargement de {img_path}: {e}")
                continue

            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)

            # Calcul des prédictions
            if target_task is not None:
                output = outputs[target_task]
                probabilities = torch.softmax(output, dim=1)
                max_prob, pred_idx = torch.max(probabilities, dim=1)
                pred_idx = pred_idx.item()
                max_prob = max_prob.item()
                predicted_class = tasks[target_task][pred_idx] if pred_idx < len(tasks[target_task]) else "Unknown"
                results[rel_path] = {target_task: {"predicted_class": predicted_class, "probability": max_prob}}
            else:
                image_preds = {}
                for t, output in outputs.items():
                    probabilities = torch.softmax(output, dim=1)
                    max_prob, pred_idx = torch.max(probabilities, dim=1)
                    pred_idx = pred_idx.item()
                    max_prob = max_prob.item()
                    predicted_class = tasks[t][pred_idx] if pred_idx < len(tasks[t]) else "Unknown"
                    image_preds[t] = {"predicted_class": predicted_class, "probability": max_prob}
                results[rel_path] = image_preds

            # Pour le classement, on utilise la prédiction pour folder_task
            if target_task is not None:
                key = target_task
                pred_for_folder = predicted_class
            else:
                key = folder_task
                pred_for_folder = results[rel_path][folder_task]["predicted_class"]
            predictions_by_task[key].setdefault(pred_for_folder, []).append(rel_path)

            # Extraction de la ground truth depuis la structure du dossier (si image dans un sous-dossier)
            if os.path.abspath(root) != os.path.abspath(test_folder):
                folder_name = os.path.basename(root)
                for t, class_list in tasks_to_evaluate.items():
                    gt_class = map_folder_to_class(folder_name, class_list)
                    if gt_class is not None:
                        gt_by_task[t].append(gt_class)
                        if target_task is not None:
                            pred_val = predicted_class
                        else:
                            pred_val = results[rel_path][t]["predicted_class"]
                        pred_gt_by_task[t].append(pred_val)

            # Annotation et sauvegarde de l'image
            if save_test_images:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                y0, dy = 30, 30
                if target_task is not None:
                    annotation = f"{target_task}: {results[rel_path][target_task]['predicted_class']} ({results[rel_path][target_task]['probability']:.2f})"
                else:
                    annotation = ""
                    for t, pred in results[rel_path].items():
                        annotation += f"{t}: {pred['predicted_class']} ({pred['probability']:.2f})\n"
                cv2.putText(img_cv, annotation, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if folder_task in results[rel_path]:
                    folder_label = results[rel_path][folder_task]['predicted_class']
                else:
                    folder_label = list(results[rel_path].values())[0]['predicted_class']
                dest_folder = os.path.join(annotated_base_dir, folder_label)
                os.makedirs(dest_folder, exist_ok=True)
                annotated_path = os.path.join(dest_folder, file)
                cv2.imwrite(annotated_path, img_cv)
                cv2.imshow("Prédiction", img_cv)
                cv2.waitKey(100)
    if save_test_images:
        cv2.destroyAllWindows()

    # Calcul des scores F1 pour chaque tâche évaluée (sur les images avec ground truth)
    final_results = {}
    for t in tasks_to_evaluate.keys():
        f1_dict = {}
        global_f1 = None
        if len(gt_by_task[t]) > 0 and len(pred_gt_by_task[t]) > 0:
            unique_classes = list(set(gt_by_task[t]))
            f1_scores = f1_score(gt_by_task[t], pred_gt_by_task[t], labels=unique_classes, average=None)
            f1_dict = dict(zip(unique_classes, f1_scores))
            global_f1 = f1_score(gt_by_task[t], pred_gt_by_task[t], average='weighted')
        counts = {}
        for cls in tasks_to_evaluate[t]:
            counts[cls] = len(predictions_by_task[t].get(cls, []))
        final_results[t] = {"by_class": counts, "f1_score": f1_dict, "global_f1": global_f1}

    json_path = os.path.join(save_dir, "folder_predictions.json")
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Résultats des prédictions sauvegardés dans {json_path}")

    # Si aucune tâche cible n'est spécifiée, enregistrer un second JSON contenant les prédictions complètes pour toutes les tâches,
    # en utilisant le chemin relatif des images.
    if target_task is None:
        all_pred_json_path = os.path.join(save_dir, "all_predictions.json")
        with open(all_pred_json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Prédictions complètes sauvegardées dans {all_pred_json_path}")




# -----------------------------------------------------------------------
# FONCTIONS D'ENTRAINEMENT & EVALUATION
# -----------------------------------------------------------------------
def train_model(model, train_loader, criterions_dict, optimizer, num_epochs=25, writer=None, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_samples = 0

        for batch_idx, (inputs, labels_dict) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs_dict = model(inputs)
            loss = 0.0
            batch_size = inputs.size(0)

            for task_name, criterion in criterions_dict.items():
                lbl = labels_dict[task_name]
                if lbl is not None:
                    lbl = lbl.to(device)
                    task_out = outputs_dict[task_name]
                    task_loss = criterion(task_out, lbl)
                    loss += task_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            print(
                f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / (num_samples if num_samples else 1)
        print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        if writer:
            writer.add_scalar(f"Fold_{fold}/Train/Loss", epoch_loss, epoch)

    return model


def evaluate_model(model, val_loader, criterions_dict, writer=None, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    val_loss = 0.0
    num_samples = 0

    all_preds = {task: [] for task in criterions_dict.keys()}
    all_labels = {task: [] for task in criterions_dict.keys()}

    with torch.no_grad():
        for inputs, labels_dict in val_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            outputs_dict = model(inputs)
            loss = 0.0

            for task_name, criterion in criterions_dict.items():
                lbl = labels_dict[task_name]
                if lbl is not None:
                    lbl = lbl.to(device)
                    task_out = outputs_dict[task_name]
                    task_loss = criterion(task_out, lbl)
                    loss += task_loss

                    _, preds = torch.max(task_out, dim=1)
                    all_preds[task_name].extend(preds.cpu().numpy())
                    all_labels[task_name].extend(lbl.cpu().numpy())

            val_loss += loss.item() * batch_size
            num_samples += batch_size

    mean_loss = val_loss / (num_samples if num_samples else 1)

    metrics_dict = {}
    for task_name in criterions_dict.keys():
        if len(all_preds[task_name]) == 0:
            metrics_dict[task_name] = {'accuracy': None, 'precision': None, 'recall': None}
            continue

        preds_np = np.array(all_preds[task_name])
        labels_np = np.array(all_labels[task_name])
        accuracy = (preds_np == labels_np).mean()
        precision = precision_score(labels_np, preds_np, average='weighted', zero_division=0)
        recall = recall_score(labels_np, preds_np, average='weighted', zero_division=0)
        metrics_dict[task_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall}

    accuracies = [m['accuracy'] for m in metrics_dict.values() if m['accuracy'] is not None]
    global_acc = float(np.mean(accuracies)) if accuracies else 0.0

    print(f'Fold {fold}, Val Loss: {mean_loss:.4f}, Mean Accuracy: {global_acc:.4f}')
    for task_name, m in metrics_dict.items():
        if m['accuracy'] is not None:
            print(
                f"  Task {task_name} -> Accuracy: {m['accuracy']:.4f}, Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")
        else:
            print(f"  Task {task_name} -> Pas de labels valides dans ce fold.")

    if writer:
        writer.add_scalar(f"Fold_{fold}/Validation/Loss", mean_loss)
        writer.add_scalar(f"Fold_{fold}/Validation/MeanAccuracy", global_acc)
        for task_name, m in metrics_dict.items():
            if m['accuracy'] is not None:
                writer.add_scalar(f"Fold_{fold}/Validation/{task_name}_Accuracy", m['accuracy'])
                writer.add_scalar(f"Fold_{fold}/Validation/{task_name}_Precision", m['precision'])
                writer.add_scalar(f"Fold_{fold}/Validation/{task_name}_Recall", m['recall'])

    return mean_loss, global_acc, metrics_dict



def watch_folder_predictions(model, tasks, watch_folder, transform, device, save_dir,save_dir_to_canon, poll_interval=5):
    """
    Surveille en continu un dossier (watch_folder) contenant des images nommées avec un timestamp
    (ex: "2025-03-12_09-54-01.jpg"). À chaque nouvelle image détectée (la plus récente),
    le modèle est appliqué et :
      - Un fichier JSON "last_prediction.json" est mis à jour avec la prédiction de cette image (écrasant la précédente).
      - Un historique des prédictions est mis à jour dans un DataFrame, puis sauvegardé en CSV ("prediction_history.csv").

    Args:
        model (torch.nn.Module): Le modèle multi-tâches chargé.
        tasks (dict): Dictionnaire associant chaque tâche à la liste de ses classes.
        watch_folder (str): Dossier à surveiller.
        transform: Transformation à appliquer aux images.
        device: Appareil (CPU ou GPU).
        save_dir (str): Dossier où enregistrer les sorties (JSON et CSV).
        poll_interval (int, optional): Intervalle de sondage en secondes.
    """
    # Charger (ou créer) l'historique des prédictions dans un DataFrame
    history_file = os.path.join(save_dir, "prediction_history.csv")
    # Définir les colonnes : timestamp + pour chaque tâche, predicted_class et probability
    columns = ["timestamp"]
    for t, class_list in tasks.items():
        columns.extend([f"{t}_predicted_class", f"{t}_probability"])
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
    else:
        history_df = pd.DataFrame(columns=columns)

    last_processed = None  # pour mémoriser le dernier fichier traité

    print(f"Surveillance du dossier {watch_folder} toutes les {poll_interval} secondes...")
    while True:
        files = [f for f in os.listdir(watch_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not files:
            time.sleep(poll_interval)
            continue

        # On suppose que les noms suivent le format ISO "YYYY-MM-DD_HH-MM-SS.ext"
        files.sort()
        last_file = files[-1]
        if last_file == last_processed:
            time.sleep(poll_interval)
            continue
        last_processed = last_file

        full_path = os.path.join(watch_folder, last_file)
        try:
            img = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Erreur lors du chargement de {full_path}: {e}")
            time.sleep(poll_interval)
            continue

        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)

        prediction = {}
        # Pour chaque tâche, extraire la prédiction
        for t, output in outputs.items():
            probabilities = torch.softmax(output, dim=1)
            max_prob, pred_idx = torch.max(probabilities, dim=1)
            pred_idx = pred_idx.item()
            max_prob = max_prob.item()
            predicted_class = tasks[t][pred_idx] if pred_idx < len(tasks[t]) else "Unknown"
            prediction[t] = {"predicted_class": predicted_class, "probability": max_prob}

        # Extraire le timestamp à partir du nom de fichier (en supprimant l'extension)
        timestamp_str = os.path.splitext(last_file)[0]  # ex: "2025-03-12_09-54-01"
        try:
            timestamp_dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        except Exception as e:
            print(f"Erreur de parsing du timestamp pour {last_file}: {e}")
            timestamp_dt = datetime.datetime.now()
            timestamp_str = timestamp_dt.strftime("%Y-%m-%d_%H-%M-%S")

        # Enregistrer le JSON de la dernière prédiction (écrase le précédent)
        last_pred_json = os.path.join(save_dir, "last_prediction.json")
        with open(last_pred_json, "w") as f:
            json.dump({"timestamp": timestamp_str, "prediction": prediction}, f, indent=4)
        print(f"Prédiction de {last_file} enregistrée dans {last_pred_json}")

        if save_dir_to_canon != None:
            # Enregistrer le JSON de la dernière prédiction (écrase le précédent)
            last_pred_json = os.path.join(save_dir_to_canon, "WeatherInfos.json")
            with open(last_pred_json, "w") as f:
                json.dump({"timestamp": timestamp_str, "prediction": prediction}, f, indent=4)
            print(f"Prédiction de {last_file} enregistrée dans {last_pred_json}")


        # Mettre à jour l'historique (ajouter une ligne dans le DataFrame)
        row = {"timestamp": timestamp_str}
        for t, pred in prediction.items():
            row[f"{t}_predicted_class"] = pred["predicted_class"]
            row[f"{t}_probability"] = pred["probability"]
        # Utilisation de pd.concat à la place de .append pour éviter l'erreur
        history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
        # Sauvegarder l'historique en CSV
        history_df.to_csv(history_file, index=False)
        print(f"Historique mis à jour : {history_file}")

        time.sleep(poll_interval)

