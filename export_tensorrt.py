from ultralytics import YOLO

def compilar_modelo():
    print("Iniciando a conversão para TensorRT... Isso pode demorar alguns minutos.")
    
    # 1. Carrega o modelo original que está no seu Config.py
    model = YOLO("models/yolo26n.pt")
    
    # 2. Exporta para o formato 'engine'
    # imgsz: Fixar o tamanho da imagem otimiza ainda mais (usamos os valores do seu Config.py)
    # half: Usa precisão FP16 (metade da memória, dobro da velocidade)
    # workspace: Permite que o TensorRT use até 4GB de VRAM durante a compilação para buscar a melhor rota matemática
    model.export(
        format="engine",
        device=0,
        half=True,
        imgsz=(480, 640), 
        workspace=4
    )
    
    print("Conversão concluída! Verifique a pasta 'models/' pelo arquivo .engine")

if __name__ == "__main__":
    compilar_modelo()