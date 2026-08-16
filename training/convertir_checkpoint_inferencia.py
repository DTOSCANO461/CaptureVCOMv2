#!/usr/bin/env python3
"""convertir_checkpoint_inferencia.py -- convierte un checkpoint de VideoMAEv1
(state_dict guardado por train_binario.py) al formato de bias de atencion
"viejo" (q_bias/v_bias, sin key.bias) que usa transformers<5.x.

Por que hace falta: transformers>=5.x separo las bias de atencion en
query.bias/key.bias/value.bias (3 parametros entrenables independientes,
key.bias no existe en la arquitectura vieja). Un checkpoint entrenado con
transformers>=5.x (ej. en la VM, ver train_binario.py::_remapea_bias_atencion)
tiene esas 3 keys por capa; un entorno con transformers<5.x (ej. yolo2,
4.57.1, usado para inferencia) espera q_bias/v_bias -- load_state_dict
fallaria/dejaria pesos sin cargar si se intenta usar el .pt tal cual.

Validado en vivo (2026-08-15): cargar un checkpoint asi convertido (con
key.bias simplemente DESCARTADO -- la arquitectura vieja no tiene donde
ponerlo) y evaluarlo contra el set de validacion completo (9317 tubos) dio
EXACTAMENTE el mismo val_auc/val_ap que el reportado durante el
entrenamiento original -- la magnitud de key.bias aprendida (no cero, hasta
norma ~3.5 en capas profundas) no tiene impacto medible en las predicciones
finales del clasificador binario.

Si el checkpoint ya esta en formato viejo (entrenado con transformers<5.x,
ej. localmente), este script lo detecta y no hace nada -- es seguro
correrlo sobre cualquier checkpoint sin saber de antemano en que entorno
se entreno.

Uso:
  python3 training/convertir_checkpoint_inferencia.py entrada.pt salida.pt
"""
import sys
import torch


def convertir(sd):
    tiene_formato_nuevo = any(k.endswith("attention.attention.key.bias") for k in sd)
    if not tiene_formato_nuevo:
        return sd, 0  # ya esta en formato viejo, nada que hacer

    nuevo = {}
    n_convertidas = 0
    for k, v in sd.items():
        if k.endswith("attention.attention.key.bias"):
            continue  # no existe en el formato viejo, se descarta (validado sin impacto medible)
        elif k.endswith("attention.attention.query.bias"):
            nuevo[k.replace("query.bias", "q_bias")] = v
            n_convertidas += 1
        elif k.endswith("attention.attention.value.bias"):
            nuevo[k.replace("value.bias", "v_bias")] = v
            n_convertidas += 1
        else:
            nuevo[k] = v
    return nuevo, n_convertidas


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    entrada, salida = sys.argv[1], sys.argv[2]
    sd = torch.load(entrada, map_location="cpu", weights_only=True)
    nuevo, n = convertir(sd)
    if n == 0:
        print(f"{entrada}: ya esta en formato viejo (o no es VideoMAEv1), copiado tal cual.")
    else:
        print(f"{entrada}: {n} bias convertidas (formato nuevo -> viejo, key.bias descartado, "
              f"validado sin impacto medible en val_auc/val_ap).")
    torch.save(nuevo, salida)
    print(f"-> {salida}")


if __name__ == "__main__":
    main()
