import os
import shutil

class DatasetDownloader:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.available_datasets = {
            "malaria": "kaggle datasets download shahriar26s/malaria-detection",
            "covid": "kaggle datasets download jongallegos/dataset-for-cs-770"
        }

    def download(self, ds_nomi=None):
        assert ds_nomi in self.available_datasets, f"Mavjud bo'lgan datasetlardan birini tanlang: {list(self.available_datasets.keys())}"

        dataset_path = os.path.join(self.save_dir, ds_nomi)
        if os.path.isfile(f"{dataset_path}.csv") or os.path.isdir(dataset_path):
            print(f"Dataset allaqachon mavjud: {dataset_path}")
            return

        url = self.available_datasets[ds_nomi]
        if not url:
            print(f"{ds_nomi} dataset uchun yuklab olish manzili yo'q.")
            return

        dataset_folder_name = url.split("/")[-1]
        full_path = os.path.join(self.save_dir, dataset_folder_name)

        print(f"{ds_nomi} dataset yuklanmoqda...")
        os.system(f"{url} -p {full_path}")

        archive_path = os.path.join(full_path, f"{dataset_folder_name}.zip")
        extracted_path = os.path.join(self.save_dir, dataset_folder_name)

        if os.path.exists(archive_path):
            shutil.unpack_archive(archive_path, extracted_path)
            os.remove(archive_path)
            os.rename(extracted_path, dataset_path)
            print(f"{ds_nomi} dataset '{dataset_path}' ga muvaffaqiyatli yuklandi!")
        else:
            print("Arxiv fayl topilmadi, ehtimol yuklab olish muvaffaqiyatsiz bo'lgan.")