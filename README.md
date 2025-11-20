# CAN KAN?

**Cankan** es un entorno de experimentación e implementación basado en **Kolmogorov–Arnold Networks (KAN)**. Este repositorio realiza experimentos para comparar una KAN con una MLP usando diferentes optimizadores.

El proyecto toma como base el paper fundacional de Ximming Liu: *"KAN: Kolmogorov–Arnold Networks"* ([arXiv:2404.19756](https://arxiv.org/abs/2404.19756)).


---

##  Instalación

Se recomienda el uso de **Conda** para gestionar el entorno, asegurando compatibilidad con Python 3.12.

### 1. Crear el entorno virtual
```bash
conda create -n cankan python=3.12 -y
```

### 2. Activar e instalar dependencias
```bash
conda activate cankan
pip install -r requirements.txt
```

---

##  Estructura del Proyecto


```text
cankan/
├── kan/                       
│   ├── MultKAN.py              
│   ├── KANLayer.py             
│   ├── Symbolic_KANLayer.py    
│   ├── compiler.py            
│   └── hypothesis.py           
│
├── efficient_kan/              
│   ├── efficient_kan.py        
│   └── mnist.py                
├── src/                        
│   ├── run.py                  
│   ├── mlp.py                  
│   └── model/                  
│
├── notebooks/                  
│   └── (Explicaciones, Experimentos)
│
└── figures/                    
```

---

## ⚡ Uso Básico

### Ejecutar un experimento
El script `src/run.py` es el punto de entrada principal para entrenar modelos, generar datos sintéticos y visualizar resultados.

```bash
python src/run.py
```

### Exploración Interactiva
Para entender cómo funcionan los B-Splines o la regresión simbólica, navega a la carpeta `notebooks/`:

```bash
jupyter notebook notebooks/
```

---

## Reporte técnico

Revisa el reporte técnico en: 