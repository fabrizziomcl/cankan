# CANKAN (KAN experiments & utilities)

Resumen corto
- Implementación experimental de Kolmogorov–Arnold Networks (KAN), utilidades para descubrimiento simbólico, poda/refinamiento de grids y comparaciones con MLPs.
- Basado en la línea de trabajo de KAN — ver el paper original: Liu et al., "KAN: Kolmogorov–Arnold Networks" (arXiv: https://arxiv.org/abs/2404.19756).

Instalación (recomendada: conda)
1. Crear el entorno (Python 3.12):
   conda create -n cankan python=3.12 -y
2. Activar e instalar dependencias (ejemplo mínimo):
   conda activate cankan
   pip install -r requirements.txt

Estructura principal (destacados)
- kan/ — implementación central
  - [`kan/MultKAN.py`](kan/MultKAN.py) — modelo principal: [`kan.MultKAN.MultKAN`](kan/MultKAN.py)
  - [`kan/KANLayer.py`](kan/KANLayer.py) — definiciones de splines/grids: [`kan.KANLayer.KANLayer`](kan/KANLayer.py)
  - [`kan/compiler.py`](kan/compiler.py) — compilador expr → MultKAN: `expr2kan`
  - [`kan/Symbolic_KANLayer.py`](kan/Symbolic_KANLayer.py) — capa simbólica / integración SymPy
  - [`kan/hypothesis.py`](kan/hypothesis.py) — detección de separabilidad / análisis jerárquico
- efficient_kan/ — implementación ligera / alternativa
  - [`efficient_kan/efficient_kan.py`](efficient_kan/efficient_kan.py) — [`efficient_kan.KAN`](efficient_kan/efficient_kan.py)
  - [`efficient_kan/mnist.py`](efficient_kan/mnist.py) — script MNIST de ejemplo
- src/ — scripts y runners
  - [`src/run.py`](src/run.py) — runner de experiments (generación de datos, entrenamiento, plots)
  - [`src/mlp.py`](src/mlp.py) — MLP de referencia usada en benchmarks
  - [`src/model/`](src/model/) — checkpoints y estados guardados
- notebooks/ — notebooks didácticos (splines, stacked splines, grids, symbolic learning, Rosenbrock)
- test/, figures/ — resultados, figuras y tests/ejemplos guardados

Report en [reporte].