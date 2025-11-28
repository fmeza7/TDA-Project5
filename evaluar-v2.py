import sys
import os.path

class Deteccion:
    def __init__(self, id_deteccion, television, desde, largo, comercial, score):
        self.id_deteccion = id_deteccion
        self.television = television
        self.desde = desde
        self.largo = largo
        self.comercial = comercial
        self.score = score

    def interseccion(self, otra):
        if self.television != otra.television or self.comercial != otra.comercial:
            return 0
        ini1 = self.desde
        end1 = self.desde + self.largo
        ini2 = otra.desde
        end2 = otra.desde + otra.largo
        inter = min(end1, end2) - max(ini1, ini2)
        union = max(end1, end2) - min(ini1, ini2)
        if inter <= 0 or union <= 0:
            return 0
        return inter / union
    
    def toString(self):
        return "{} {} {} {} {}".format(self.television, self.desde, self.largo, self.comercial, self.score)

def get_videoname(filepath):
    name = filepath.lower().strip()
    if name.rfind('/') >= 0:
        name = name[name.rfind('/') + 1:]
    if name.rfind('\\') >= 0:
        name = name[name.rfind('\\') + 1:]
    if name.endswith(".mp4") or name.endswith(".mpg"):
        name = name[0:-4]
    return name

def leer_archivo_detecciones(filename, with_scores):
    if not os.path.isfile(filename):
        raise Exception("no existe el archivo {}".format(filename))
    detecciones = []
    cont_lineas = 0
    with open(filename) as f:
        for linea in f:
            cont_lineas += 1
            try:
                linea = linea.rstrip("\r\n")
                if linea == "" or linea.startswith("#"):
                    continue
                partes = linea.split("\t")
                if with_scores and len(partes) != 5:
                    raise Exception("incorrecto numero de columnas (se esperan 5 columnas separadas por un tabulador")
                if not with_scores and len(partes) != 4:
                    raise Exception("incorrecto numero de columnas (se esperan 4 columnas separadas por un tabulador")
                #nombre de video (sin ruta ni extension)
                television = get_videoname(partes[0])
                if television == "":
                    raise Exception("nombre television invalido")
                #nombre de comercial (sin ruta ni extension)
                comercial = get_videoname(partes[3])
                if comercial == "":
                    raise Exception("nombre comercial invalido")
                #los tiempos pueden incluir milisegundos
                desde = round(float(partes[1]), 3)
                if desde < 0:
                    raise Exception("valor incorrecto desde={}".format(desde))
                largo = round(float(partes[2]), 3)
                if largo <= 0:
                    raise Exception("valor incorrecto largo={}".format(largo))
                #score
                score = 0
                if with_scores:
                    score = float(partes[4])
                    if score <= 0:
                        raise Exception("valor incorrecto score={}".format(score))
                det = Deteccion(cont_lineas, television, desde, largo, comercial, score)
                detecciones.append(det)
            except Exception as ex:
                print ("Error {} (linea {}): {}".format(filename, cont_lineas, ex))
    print ("{} detecciones en archivo {}".format(len(detecciones), filename))
    return detecciones

class ResultadoDeteccion:
    def __init__(self, deteccion):
        self.deteccion = deteccion
        self.es_incorrecta = False
        self.es_repetida = False
        self.es_correcta = False
        self.gt = None
        self.iou = 0

class Metricas:
    def __init__(self, threshold):
        self.threshold = threshold
        self.total_gt = 0
        self.correctas = 0
        self.incorrectas = 0
        self.repetidas = 0
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.iou = 0
        self.metrica_tarea = 0

class Evaluacion:
    def load_detecciones(self, filename_detecciones):
        #cargar las detecciones
        self.detecciones = leer_archivo_detecciones(filename_detecciones, True)

    def load_gt(self, filename_gt):
        #cargar el ground-truth
        self.groundtruth = leer_archivo_detecciones(filename_gt, False)

    def evaluar_detecciones(self):
        #ordenar detecciones por score
        self.detecciones.sort(key=lambda x: x.score, reverse=True)
        #seleccionar del ground-truth solo los videos de television de las detecciones
        videos_tv = set()
        for det in self.detecciones:
            videos_tv.add(det.television)
        #filtrar gt relevante
        self.detecciones_gt = []
        for gt in self.groundtruth:
            if gt.television in videos_tv:
                self.detecciones_gt.append(gt)
        #para descartar las detecciones duplicadas
        ids_encontradas = set()
        #revisar cada deteccion
        self.resultados = []
        for det in self.detecciones:
            #evaluar si la deteccion es correcta a no
            res = self.buscar_deteccion_en_gt(det, ids_encontradas)
            self.resultados.append(res)
        #ordenar los resultados como el archivo de entrada
        self.resultados.sort(key=lambda x: x.deteccion.id_deteccion)

    def buscar_deteccion_en_gt(self, deteccion, ids_encontradas):
        gt_encontrada = None
        best_iou = 0
        #busca en gt la deteccion que tiene mayor interseccion
        for det_gt in self.detecciones_gt:
            inter = deteccion.interseccion(det_gt)
            if inter > best_iou:
                gt_encontrada = det_gt
                best_iou = inter
        #retorna resultado
        res = ResultadoDeteccion(deteccion)
        if gt_encontrada is None:
            res.es_incorrecta = True
        elif gt_encontrada.id_deteccion in ids_encontradas:
            res.es_repetida = True
        else:
            res.es_correcta = True
            res.gt = gt_encontrada
            res.iou = best_iou
            ids_encontradas.add(gt_encontrada.id_deteccion)
        return res

    def imprimir_resultados(self):
        if len(self.resultados) == 0:
            return
        print ("Resultados {} detecciones:".format(len(self.resultados)))
        for res in self.resultados:
            s1 = ""
            s2 = ""
            if res.es_correcta:
                s1 = "OK"
                s2 = "  //IoU={}% (gt={} {})".format(round(100 * res.iou), res.gt.desde, res.gt.largo)
            elif res.es_repetida:
                s1 = "dp"
            elif res.es_incorrecta:
                s1 = "--"
            print(" {}) {}{}".format(s1, res.deteccion.toString(), s2))
        
    def get_all_scores(self):
        scores = set()
        for res in self.resultados:
            if res.es_correcta:
                scores.add(res.deteccion.score)
        scores.add(0)
        return sorted(list(scores), reverse=True)

    def calcular_metricas(self, score_threshold):
        met = Metricas(score_threshold)
        met.total_gt = len(self.detecciones_gt)
        suma_iou = 0
        for res in self.resultados:
            #ignorar detecciones con score bajo el umbral
            if res.deteccion.score < score_threshold:
                continue
            if res.es_correcta:
                met.correctas += 1
                suma_iou += res.iou
            if res.es_incorrecta:
                met.incorrectas += 1
            if res.es_repetida:
                met.repetidas += 1
        if met.correctas > 0:
            met.recall = met.correctas / met.total_gt
            met.precision = met.correctas / (met.correctas + met.incorrectas)
        if met.precision > 0 and met.recall > 0:
            met.f1 = (2 * met.precision * met.recall) / (met.precision + met.recall)
        if met.correctas > 0:
            met.iou = suma_iou / met.correctas
        if met.correctas > 0 and met.correctas > met.incorrectas:
            met.metrica_tarea = (met.correctas - met.incorrectas) / met.total_gt
        return met

def evaluar(filename_detecciones, filename_gt):
    ev = Evaluacion()
    ev.load_gt(filename_gt)
    ev.load_detecciones(filename_detecciones)
    ev.evaluar_detecciones()
    ev.imprimir_resultados()
    #calcular las metricas para cada score y seleccionar el mejor
    mejor = None
    for score in ev.get_all_scores():
        metricas = ev.calcular_metricas(score)
        if mejor is None or metricas.metrica_tarea > mejor.metrica_tarea:
            mejor = metricas
    if mejor is not None:
        print ("Mejor resultado usando el umbral {}:".format(mejor.threshold))
        print ("  Precision={}%  Recall={}%  F1={}  IoU={}".format(round(100 * mejor.precision, 1), round(100 * mejor.recall, 1), round(mejor.f1, 3), round(mejor.iou, 3)))
        print ("  Correctas={}  Incorrectas={}  Repetidas={}  Total_GT={}".format(mejor.correctas, mejor.incorrectas, mejor.repetidas, mejor.total_gt))
        print ("  Resultado Tarea={}%".format(round(100 * mejor.metrica_tarea, 1)))

#inicio
if len(sys.argv) < 3:
    print ("Evaluacion Tarea 1")
    print ("Uso: {} [archivo_detecciones.txt] [gt.txt]".format(sys.argv[0]))
    sys.exit(1)

filename_detecciones = sys.argv[1]
filename_gt = sys.argv[2]
evaluar(filename_detecciones, filename_gt)
