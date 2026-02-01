def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox):
    """
    Calcola la posizione dei piedi con una leggera correzione prospettica.
    Non prendiamo il pixel più basso assoluto (y2), ma un po' più su, 
    dove di solito c'è il contatto fisico centrale scarpe-pavimento.
    """
    x1, y1, x2, y2 = bbox
    # Prendiamo il centro orizzontale
    x_center = (x1 + x2) / 2
    # Correzione: alziamo il punto del 5% dell'altezza del box 
    # per centrare meglio l'ellisse sotto il corpo del giocatore in corsa
    y_ground = y2 - (y2 - y1) * 0.05 
    
    return x_center, y_ground

def check_side(p1: tuple[float, float], pc: tuple[float, float], id_p1: int, id_frame: int) -> int:
    '''
    Check side logic preserved from your code.
    '''
    left_ids  = {0,1,2,3,4,5,8,9}
    right_ids = {10,11,12,13,14,15,16,17}

    if p1[0] > pc[0]:          # a destra
        return 1 if id_p1 in right_ids else 0
    else:                      # a sinistra
        return 1 if id_p1 in left_ids else 0