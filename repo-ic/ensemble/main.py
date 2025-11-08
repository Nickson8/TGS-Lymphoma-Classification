def main():

    TGS_imgs, TGS_nomes = load_images_from_folder("tgs-imgs")
    Lymphomas_imgs, Lymphomas_nomes = load_images_from_folder("linfoma-imgs")

    print("Terminou de carregar as imagens")

    epochs = 50 #80
    batch_size = 64 #64

    X = TGS_imgs + Lymphomas_imgs
    X_nomes = TGS_nomes + Lymphomas_nomes
    y = [0]*len(TGS_imgs) + [1]*len(Lymphomas_imgs)

    total = Total_Writer_Ensemble(batch_size=batch_size, lr=1e-3)
    #Treina, printa os logs e salva os folds
    total.treina(X, y, epochs)
    total.gera_relatorios(X, X_nomes, y)


if __name__ == "__main__":
    main()