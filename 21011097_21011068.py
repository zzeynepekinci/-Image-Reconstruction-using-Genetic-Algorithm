

from PIL import Image
import numpy as np
import pygad
import matplotlib.pyplot as plt

def load_binary_image_png(path):
    img = Image.open(path).convert('L')
    arr = np.array(img)
    arr = (arr > 128).astype(int)  # Piksel değerleri binary hale getiriliyor 128'den büyükse 1, değilse 0
    return arr 

def find_best_pattern(block, patterns):
    min_dist = float('inf')
    best = None
    for pattern in patterns:
        dist = np.sum(np.abs(block - pattern)) #pattern ile blok arasindaki fark hesaplanir (manhattan dist)
        if dist < min_dist: #bulunan sonuc mevcut en iyi sonuctan kucukse en iyisi olarak isaretlenir.
            min_dist = dist
            best = pattern
    return best
#patternlerle olusturulan gorseli dondurme fonksiyonu
def reconstruct_image(img, patterns):
    reconstructed = np.zeros_like(img)
    for i in range(0, 24, 3): #goruntu 3*3luk bloklara ayrilarak gezilir.
        for j in range(0, 24, 3):
            block = img[i:i+3, j:j+3] #siradaki 3*3luk blok alinir.
            best = find_best_pattern(block, patterns) # patternler arasinda bu bloga en uygun olan pattern bulunur.
            reconstructed[i:i+3, j:j+3] = best #yeniden olusturulan 24*24luk gorsele bu blok eklenir.
    return reconstructed #patternlerle olusturulan gorsel dondurulur.

def plot_patterns(patterns, title): # patternleri ekrana yazdiran fonksiyon
    fig, axs = plt.subplots(1, 7, figsize=(12, 2))
    for i, pattern in enumerate(patterns):
        axs[i].imshow(pattern, cmap='gray')
        axs[i].set_title(f"P{i+1}")
        axs[i].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_all_reconstructions(images, patterns, deney_etiketi=""): #orijinal gorselii ve onun pattern'larla yeniden oluşturulmuş halini yan yana gösteren fonksiyondur
    plt.figure(figsize=(10, 10))
    for idx, img in enumerate(images):
        recon = reconstruct_image(img, patterns)

        plt.subplot(len(images), 2, 2*idx+1)
        plt.title(f"Orijinal Resim {idx+1}")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(len(images), 2, 2*idx+2)
        plt.title(f"Yeniden Oluşturulan {idx+1} - {deney_etiketi}")
        plt.imshow(recon, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def run_experiment(sol_per_pop, mutation_percent, image_paths, dataset_label):
    images = [load_binary_image_png(p) for p in image_paths] #dosya yollarindan 5 binary gorsel yuklenir.
    total_pixels = len(images) * 24 * 24

    loss_history = []  #Her jenerasyondaki loss  değeri burada saklanacak
    similarity_history = []#Her jenerasyondaki  benzerlik değeri burada saklanacak

    def on_generation(ga_instance):
        best_solution, best_fitness, _ = ga_instance.best_solution()#O jenerasyondaki en iyi çözüm ve fitness değeri alınır.
        current_loss = -best_fitness#Fitness negatif olduğu için minimize etmek amacıyla, tekrar pozitif loss değeri elde edilir.
        current_similarity = (total_pixels - current_loss) / total_pixels#benzerlik oranı hesaplanir.
        loss_history.append(current_loss)#Her jenerasyonun sonuçları grafikte kullanmak için listeye kaydedilir.
        similarity_history.append(current_similarity)

    def fitness_func(ga_instance, solution, solution_idx):
        patterns = np.reshape(solution, (7, 3, 3)).round().astype(int)
        total_loss = 0
        for img in images: #Her 3x3 blok için en yakın pattern bulunur ve bu eşleşmedeki farklar (loss) toplanır.
            for i in range(0, 24, 3):
                for j in range(0, 24, 3):
                    block = img[i:i+3, j:j+3]
                    best = find_best_pattern(block, patterns)
                    total_loss += np.sum(np.abs(block - best))
        return -total_loss #Genetik algoritma maksimizasyon yaptığı için negatif loss döndürülür(lossu kücültebilmek icin)

    ga_instance = pygad.GA(
        num_generations=100, #jenerasyon sayisi ayarlanir.
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=63,
        gene_type=int,
        gene_space=[0, 1],
        parent_selection_type="tournament", #tournament yöntemi secilir.
        crossover_type="single_point",#caprazlama tek noktadan yapilir
        mutation_type="random",#mutasyon random olarak gerceklesir
        mutation_percent_genes=mutation_percent,
        on_generation=on_generation,
        keep_parents=2 #en iyi 2 birey bir sonraki nesle tasinir.
    )

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    final_loss = -solution_fitness
    final_similarity = (total_pixels - final_loss) / total_pixels
    best_patterns = np.reshape(solution, (7, 3, 3)) #en iyi patternlar elde ediliyor.

    deney_etiketi = f"{dataset_label} | Pop:{sol_per_pop}, Mutasyon:%{mutation_percent}"
    print(f"\n=== {deney_etiketi} ===")
    print(f"Final Loss: {final_loss}")
    print(f"Benzerlik Oranı: %{final_similarity * 100:.2f}")

    show_all_reconstructions(images, best_patterns, deney_etiketi)
    plot_patterns(best_patterns, f"{dataset_label} - En İyi Patternler")

    return loss_history, similarity_history, deney_etiketi


if __name__ == "__main__":
    dataset1 = ["binary_gorsel1.png", "binary_gorsel2.png", "binary_gorsel3.png",
                "binary_gorsel4.png", "binary_gorsel5.png"]

    dataset2 = [f"dataset2/img_{i}.png" for i in range(1, 6)]
    dataset3 = [f"dataset3/img_{i}.png" for i in range(1, 6)]

    all_datasets = [
    #("Küme 1", dataset1),
       # ("Küme 2", dataset2),
        ("Küme 3", dataset3)
    ]

    experiments = [
        {"pop": 5, "mutation": 1},
        {"pop": 10, "mutation": 1},
       {"pop": 15, "mutation": 1},
    ]

    for dataset_label, dataset_paths in all_datasets:
        all_loss_histories = []
        all_similarity_histories = []
        labels = []

        for exp in experiments:
            pop = exp["pop"]
            mutation = exp["mutation"]

            loss_hist, sim_hist, label = run_experiment(
                sol_per_pop=pop,
                mutation_percent=mutation,
                image_paths=dataset_paths,
                dataset_label=dataset_label
            )
            all_loss_histories.append(loss_hist)
            all_similarity_histories.append(sim_hist)
            labels.append(f"Pop:{pop}, Mutasyon:%{mutation}")

        
        plt.figure(figsize=(10, 4))
        for i in range(len(all_loss_histories)):
            plt.plot(all_loss_histories[i], label=labels[i])
        plt.xlabel("Jenerasyon")
        plt.ylabel("Toplam Loss")
        plt.title(f"Loss Karşılaştırması - {dataset_label}")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 4))
        for i in range(len(all_similarity_histories)):
            plt.plot(all_similarity_histories[i], label=labels[i])
        plt.xlabel("Jenerasyon")
        plt.ylabel("Benzerlik Oranı")
        plt.title(f"Benzerlik Karşılaştırması - {dataset_label}")
        plt.legend()
        plt.grid()
        plt.show()





