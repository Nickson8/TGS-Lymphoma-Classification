def main():

    TGS_imgs = load_images_from_folder("tgs-imgs")
    Lymphomas_imgs = load_images_from_folder("linfoma-imgs")

    print("Terminou de carregar as imagens")

    epochs = 2 #80
    batch_size = 64 #64

    X = TGS_imgs + Lymphomas_imgs
    y = [0]*len(TGS_imgs) + [1]*len(Lymphomas_imgs)

    total = Total_Writer_Ensemble(MetaLearner(), batch_size=batch_size, lr=1e-3)
    #Treina, printa os logs e salva os folds
    total.treina(X, y, epochs)
    total.gera_relatorios(X, y)


if __name__ == "__main__":
    main()