from pyomo.environ import *
from pyomo.common.timing import TicTocTimer
from time import process_time
import numpy as np
import matplotlib.pyplot as plt

# =======================================================================================
# SPARSE VARIABLE VERSION - MODEL WITH MANUFACTURING QUEUE
# 
# This is the optimized version using sparse index sets.
# Mathematically equivalent to the original but with far fewer variables.
# =======================================================================================

model = AbstractModel()
data = 'Data files/Data2000_profileA_clustered.dat'
t1_start = process_time()

# ============================================================================
# MODEL WITH MANUFACTURING QUEUE
# ============================================================================
# This model allows samples to wait at the manufacturing site for up to 
# QMAX time periods before starting manufacturing. This enables smoothing
# of capacity utilization to stay below a maximum threshold (UTIL_MAX).
# ============================================================================

# SETS
model.c = Set() # Leukapheresis sites
model.h = Set() # Hospitals
model.j = Set() # Transport mode
model.m = Set() # Manufacturing sites
model.p = Set() # Patients
model.t = RangeSet(130) # Time

# Indexed PARAMETERS
model.CIM = Param(model.m) # Capital investment for manufacturing facility
model.FCAP = Param(model.m) # Total capacity of manufacturing site
model.TT1 = Param(model.j) # Transport time LS to MS using transport mode j
model.TT3 = Param(model.j) # Transport time MS to hospital using transport mode j
model.U1 = Param(model.c, model.m, model.j) # Unit transport cost LS to MS using transport mode j
model.U3 = Param(model.m, model.h, model.j) # Unit transport cost MS to hospital using transport mode j
model.INC = Param(model.p, model.c, model.t, initialize=0) # Demand therapy p arriving for leukapheresis at LS site c at time t
model.CVM = Param(model.m, default={'m1':20920, 'm2':156900, 'm3':52300, 'm4':20920, 'm5':156900, 'm6':52300, 'm7': 313800}) #Fixed variable costs


# Scalar PARAMETERS
model.FMAX = Param() # Maximum flow
model.FMIN = Param() # Minimum flow
model.TAD = Param(within=NonNegativeReals) # Duration of administration
model.TLS = Param(within=NonNegativeReals) # Duration of leukapheresis
model.TMFE = Param(default=7) # Duration of manufacturing excluding QC
model.TQC = Param(default=7) # Duration of QC
model.C_material = Param(default=10476) # Materials cost per therapy
model.CQC= Param(default=9312) # QC cost per therapy
model.ND = Param(default = 23) # Maximum turnaround (vein-to-vein) time

# NEW QUEUE PARAMETERS
model.QMAX = Param(default=5) # Maximum queue time at manufacturing site (days sample can wait)
model.UTIL_MAX = Param(default=0.75) # Maximum utilization allowed (e.g., 0.75 = 75%)


# =======================================================================================
# SPARSE INDEX SETS - Built after data is loaded
# =======================================================================================

def build_patient_data_rule(model):
    """
    After data loads, compute each patient's origin (c, t_arrival) and destination h.
    """
    c_to_h = {'c1': 'h1', 'c2': 'h2', 'c3': 'h3', 'c4': 'h4'}
    
    model.patient_origin_c = {}
    model.patient_origin_t = {}
    model.patient_dest_h = {}
    
    for p in model.p:
        for c in model.c:
            for t in model.t:
                if value(model.INC[p, c, t]) > 0:
                    model.patient_origin_c[p] = c
                    model.patient_origin_t[p] = t
                    model.patient_dest_h[p] = c_to_h[c]
                    break
            if p in model.patient_origin_c:
                break

model.BuildPatientData = BuildAction(rule=build_patient_data_rule)


# Sparse index set for Y1: only (p, c, m, j, t) where c is patient's origin
def Y1_index_init(model):
    for p in model.p:
        if p in model.patient_origin_c:
            c = model.patient_origin_c[p]
            for m in model.m:
                for j in model.j:
                    for t in model.t:
                        yield (p, c, m, j, t)

model.Y1_index = Set(dimen=5, initialize=Y1_index_init)


# Sparse index set for Y2: only (p, m, h, j, t) where h is patient's destination
def Y2_index_init(model):
    for p in model.p:
        if p in model.patient_dest_h:
            h = model.patient_dest_h[p]
            for m in model.m:
                for j in model.j:
                    for t in model.t:
                        yield (p, m, h, j, t)

model.Y2_index = Set(dimen=5, initialize=Y2_index_init)


# Sparse index sets for flow variables (same structure as Y1/Y2)
def LSR_index_init(model):
    return model.Y1_index
model.LSR_index = Set(dimen=5, initialize=LSR_index_init)

def LSA_index_init(model):
    return model.Y1_index
model.LSA_index = Set(dimen=5, initialize=LSA_index_init)

def MSO_index_init(model):
    return model.Y2_index
model.MSO_index = Set(dimen=5, initialize=MSO_index_init)

def FTD_index_init(model):
    return model.Y2_index
model.FTD_index = Set(dimen=5, initialize=FTD_index_init)

def OUTC_index_init(model):
    for p in model.p:
        if p in model.patient_origin_c:
            c = model.patient_origin_c[p]
            for t in model.t:
                yield (p, c, t)
model.OUTC_index = Set(dimen=3, initialize=OUTC_index_init)

def INH_index_init(model):
    for p in model.p:
        if p in model.patient_dest_h:
            h = model.patient_dest_h[p]
            for t in model.t:
                yield (p, h, t)
model.INH_index = Set(dimen=3, initialize=INH_index_init)


# =======================================================================================
# VARIABLES - Using sparse index sets where applicable
# =======================================================================================

# Binary VARIABLES
model.E1 = Var(model.m, within=Binary)
model.X1 = Var(model.c, model.m, within=Binary)
model.X2 = Var(model.m, model.h, within=Binary)

# SPARSE binary variables
model.Y1 = Var(model.Y1_index, within=Binary)
model.Y2 = Var(model.Y2_index, within=Binary)

# NEW: Binary variable for when manufacturing starts (keep dense - needed for queue logic)
model.Z = Var(model.p, model.m, model.t, within=Binary)

# SPARSE integer variable
model.INH = Var(model.INH_index, within=NonNegativeIntegers)

# SPARSE flow variables
model.LSR = Var(model.LSR_index, within=NonNegativeReals)
model.LSA = Var(model.LSA_index, within=NonNegativeReals)
model.MSO = Var(model.MSO_index, within=NonNegativeReals)
model.FTD = Var(model.FTD_index, within=NonNegativeReals)
model.OUTC = Var(model.OUTC_index, within=NonNegativeReals)

# Dense variables (needed for queue/manufacturing logic across all m)
model.CTM = Var(model.p, within=NonNegativeReals)
model.TTC = Var(model.p, within=NonNegativeReals)
model.OUTM = Var(model.p, model.m, model.t, within=NonNegativeReals)
model.INM = Var(model.p, model.m, model.t, within=NonNegativeReals)
model.STARTM = Var(model.p, model.m, model.t, within=NonNegativeReals)
model.DURV = Var(model.p, model.m, model.t, within=NonNegativeReals)
model.RATIO = Var(model.m, model.t, within=NonNegativeReals)

# Queue variables
model.QUEUE = Var(model.p, model.m, model.t, within=NonNegativeReals)
model.QTIME = Var(model.p, within=NonNegativeReals)

# Other variables
model.CAP = Var(model.m, model.t)
model.TRT = Var(model.p)
model.ATRT = Var()
model.STT = Var(model.p)
model.CTT = Var(model.p)

# Track arrival and manufacturing start times for queue calculation
model.ARR_TIME = Var(model.p, within=NonNegativeReals)
model.MFG_START_TIME = Var(model.p, within=NonNegativeReals)


# =======================================================================================
# OBJECTIVE FUNCTION
# =======================================================================================
def obj_rule(model):
    return sum(model.CTM[p] for p in model.p) + sum(model.TTC[p] for p in model.p) + (model.C_material + model.CQC) * len(model.p)

model.obj = Objective(rule=obj_rule)


# =======================================================================================
# CONSTRAINTS
# =======================================================================================

# Manufacturing cost
def C1_rule(model, p):
    return model.CTM[p] == sum((model.E1[m] * (model.CIM[m] + model.CVM[m])) * len(model.t) / len(model.p) for m in model.m)

model.C1 = Constraint(model.p, rule=C1_rule)


# Transportation cost - SPARSE
def C2_rule(model, p):
    y1_terms = [model.Y1[idx] * model.U1[idx[1], idx[2], idx[3]] for idx in model.Y1_index if idx[0] == p]
    y2_terms = [model.Y2[idx] * model.U3[idx[1], idx[2], idx[3]] for idx in model.Y2_index if idx[0] == p]
    y1_cost = sum(y1_terms) if y1_terms else 0
    y2_cost = sum(y2_terms) if y2_terms else 0
    return model.TTC[p] == y1_cost + y2_cost

model.C2 = Constraint(model.p, rule=C2_rule)


# ============================================================================
# UTILIZATION CONSTRAINTS
# ============================================================================

def RATIOEQ_rule(model, m, t):
    return model.RATIO[m, t] == sum(model.DURV[p, m, t] / model.FCAP[m] for p in model.p)

model.RATIOEQ = Constraint(model.m, model.t, rule=RATIOEQ_rule)


# OPTIMIZED: Use range() instead of filtered tt
def MSBnew_rule(model, p, m, t):
    return model.DURV[p, m, t] == sum(model.STARTM[p, m, tt-1] - model.OUTM[p, m, tt] for tt in range(2, t+1)) + model.OUTM[p, m, t]

model.MSBnew = Constraint(model.p, model.m, model.t, rule=MSBnew_rule)


# Maximum utilization constraint
def UTIL_CAP_rule(model, m, t):
    return model.RATIO[m, t] <= model.UTIL_MAX

model.UTIL_CAP = Constraint(model.m, model.t, rule=UTIL_CAP_rule)


# ============================================================================
# MATERIAL FLOW BALANCES - OPTIMIZED with direct indexing
# ============================================================================

# MSB1 - OPTIMIZED: Direct indexing, sparse
def MSB1_rule(model, p, c, t):
    if p not in model.patient_origin_c or model.patient_origin_c[p] != c:
        return Constraint.Skip
    
    TLS_val = value(model.TLS)
    t_out = t + TLS_val
    if t_out <= 130 and (p, c, t_out) in model.OUTC_index:
        return model.INC[p, c, t] == model.OUTC[p, c, t_out]
    elif value(model.INC[p, c, t]) == 0:
        return Constraint.Skip
    else:
        return Constraint.Skip

model.MSB1 = Constraint(model.p, model.c, model.t, rule=MSB1_rule)


# MSB3 - OPTIMIZED: Direct indexing, sparse
def MSB3_rule(model, p, c, m, j, t):
    if (p, c, m, j, t) not in model.LSR_index:
        return Constraint.Skip
    
    TT1_val = value(model.TT1[j])
    t_arrive = t + TT1_val
    if t_arrive <= 130 and (p, c, m, j, t_arrive) in model.LSA_index:
        return model.LSR[p, c, m, j, t] == model.LSA[p, c, m, j, t_arrive]
    else:
        return Constraint.Skip

model.MSB3 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=MSB3_rule)


# MSB7 - SPARSE
def MSB7_rule(model, p, c, t):
    if (p, c, t) not in model.OUTC_index:
        return Constraint.Skip
    
    lsr_terms = [model.LSR[p, c, m, j, t] for m in model.m for j in model.j if (p, c, m, j, t) in model.LSR_index]
    if len(lsr_terms) == 0:
        return model.OUTC[p, c, t] == 0
    return model.OUTC[p, c, t] == sum(lsr_terms)

model.MSB7 = Constraint(model.p, model.c, model.t, rule=MSB7_rule)


# MSB5 - SPARSE
def MSB5_rule(model, p, m, t):
    lsa_terms = [model.LSA[p, c, m, j, t] for c in model.c for j in model.j if (p, c, m, j, t) in model.LSA_index]
    if len(lsa_terms) == 0:
        return model.INM[p, m, t] == 0
    return model.INM[p, m, t] == sum(lsa_terms)

model.MSB5 = Constraint(model.p, model.m, model.t, rule=MSB5_rule)


# ============================================================================
# QUEUE CONSTRAINTS
# ============================================================================

def QUEUE_BAL_rule(model, p, m, t):
    if t == 1:
        return model.QUEUE[p, m, t] == model.INM[p, m, t] - model.STARTM[p, m, t]
    else:
        return model.QUEUE[p, m, t] == model.QUEUE[p, m, t-1] + model.INM[p, m, t] - model.STARTM[p, m, t]

model.QUEUE_BAL = Constraint(model.p, model.m, model.t, rule=QUEUE_BAL_rule)


def START_ONCE_rule(model, p):
    return sum(model.Z[p, m, t] for m in model.m for t in model.t) == 1

model.START_ONCE = Constraint(model.p, rule=START_ONCE_rule)


def STARTM_LINK_rule(model, p, m, t):
    return model.STARTM[p, m, t] == model.Z[p, m, t]

model.STARTM_LINK = Constraint(model.p, model.m, model.t, rule=STARTM_LINK_rule)


def STARTM_ARRIVAL_rule(model, p, m):
    return sum(model.STARTM[p, m, t] for t in model.t) == sum(model.INM[p, m, t] for t in model.t)

model.STARTM_ARRIVAL = Constraint(model.p, model.m, rule=STARTM_ARRIVAL_rule)


def START_AFTER_ARRIVAL_rule(model, p, m, t):
    return sum(model.STARTM[p, m, tt] for tt in range(1, t+1)) <= sum(model.INM[p, m, tt] for tt in range(1, t+1))

model.START_AFTER_ARRIVAL = Constraint(model.p, model.m, model.t, rule=START_AFTER_ARRIVAL_rule)


def ARR_TIME_rule(model, p):
    return model.ARR_TIME[p] == sum(model.INM[p, m, t] * t for m in model.m for t in model.t)

model.ARR_TIME_CALC = Constraint(model.p, rule=ARR_TIME_rule)


def MFG_START_TIME_rule(model, p):
    return model.MFG_START_TIME[p] == sum(model.STARTM[p, m, t] * t for m in model.m for t in model.t)

model.MFG_START_TIME_CALC = Constraint(model.p, rule=MFG_START_TIME_rule)


def QTIME_rule(model, p):
    return model.QTIME[p] == model.MFG_START_TIME[p] - model.ARR_TIME[p]

model.QTIME_CALC = Constraint(model.p, rule=QTIME_rule)


def QMAX_rule(model, p):
    return model.QTIME[p] <= model.QMAX

model.QMAX_CONSTRAINT = Constraint(model.p, rule=QMAX_rule)


# ============================================================================
# MSB2 - OPTIMIZED: Direct indexing
# ============================================================================
def MSB2_rule(model, p, m, t):
    TMFE_val = value(model.TMFE)
    t_out = t + TMFE_val
    if t_out <= 130:
        return model.STARTM[p, m, t] == model.OUTM[p, m, t_out]
    else:
        return Constraint.Skip

model.MSB2 = Constraint(model.p, model.m, model.t, rule=MSB2_rule)


# MSB8 - OPTIMIZED: Direct indexing, sparse
def MSB8_rule(model, p, m, t):
    TQC_val = value(model.TQC)
    t_ship = t + TQC_val
    if t_ship <= 130:
        mso_terms = [model.MSO[p, m, h, j, t_ship] for h in model.h for j in model.j if (p, m, h, j, t_ship) in model.MSO_index]
        if len(mso_terms) == 0:
            return model.OUTM[p, m, t] == 0
        return model.OUTM[p, m, t] == sum(mso_terms)
    else:
        return Constraint.Skip

model.MSB8 = Constraint(model.p, model.m, model.t, rule=MSB8_rule)


# MSB4 - OPTIMIZED: Direct indexing, sparse
def MSB4_rule(model, p, m, h, j, t):
    if (p, m, h, j, t) not in model.MSO_index:
        return Constraint.Skip
    
    TT3_val = value(model.TT3[j])
    t_arrive = t + TT3_val
    if t_arrive <= 130 and (p, m, h, j, t_arrive) in model.FTD_index:
        return model.MSO[p, m, h, j, t] == model.FTD[p, m, h, j, t_arrive]
    else:
        return Constraint.Skip

model.MSB4 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=MSB4_rule)


# MSB6 - SPARSE
def MSB6_rule(model, p, h, t):
    if (p, h, t) not in model.INH_index:
        return Constraint.Skip
    
    ftd_terms = [model.FTD[p, m, h, j, t] for m in model.m for j in model.j if (p, m, h, j, t) in model.FTD_index]
    if len(ftd_terms) == 0:
        return model.INH[p, h, t] == 0
    return model.INH[p, h, t] == sum(ftd_terms)

model.MSB6 = Constraint(model.p, model.h, model.t, rule=MSB6_rule)


# ============================================================================
# CAPACITY CONSTRAINTS - OPTIMIZED with range()
# ============================================================================
def CAP1_rule(model, m, t):
    TMFE_val = value(model.TMFE)
    return model.CAP[m, t] == model.FCAP[m] - sum(model.STARTM[p, m, tt] for p in model.p for tt in range(max(1, t - TMFE_val), t))

model.CAP1 = Constraint(model.m, model.t, rule=CAP1_rule)


def CAPCON1_rule(model, m, t):
    return sum(model.STARTM[p, m, t] for p in model.p) - sum(model.OUTM[p, m, t] for p in model.p) <= model.CAP[m, t]

model.CAPCON1 = Constraint(model.m, model.t, rule=CAPCON1_rule)


# ============================================================================
# NETWORK STRUCTURE CONSTRAINTS
# ============================================================================

#Enforce centralized mode through manufacturing site constraints
#def CON1_rule(model):
#    return sum(model.E1[m] for m in model.m) <= 2
#model.CON1 = Constraint(rule=CON1_rule)

# Force the model to use m1 and m4 specifically
#def FORCE_M1_rule(model):
#    return model.E1['m1'] == 1

#model.FORCE_M1 = Constraint(rule=FORCE_M1_rule)


#def FORCE_M4_rule(model):
#    return model.E1['m4'] == 1

#model.FORCE_M4 = Constraint(rule=FORCE_M4_rule)


def CON2_rule(model, c, m):
    return model.X1[c, m] <= model.E1[m]

model.CON2 = Constraint(model.c, model.m, rule=CON2_rule)


def CON3_rule(model, m, h):
    return model.X2[m, h] <= model.E1[m]

model.CON3 = Constraint(model.m, model.h, rule=CON3_rule)


# CON4 - SPARSE
def CON4_rule(model, p, c, m, j, t):
    if (p, c, m, j, t) not in model.Y1_index:
        return Constraint.Skip
    return model.Y1[p, c, m, j, t] <= model.X1[c, m]

model.CON4 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON4_rule)


# CON5 - SPARSE
def CON5_rule(model, p, m, h, j, t):
    if (p, m, h, j, t) not in model.Y2_index:
        return Constraint.Skip
    return model.Y2[p, m, h, j, t] <= model.X2[m, h]

model.CON5 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON5_rule)


# CON6 - SPARSE
def CON6_rule(model, p):
    y1_terms = [model.Y1[idx] for idx in model.Y1_index if idx[0] == p]
    if len(y1_terms) == 0:
        return Constraint.Infeasible
    return sum(y1_terms) == 1

model.CON6 = Constraint(model.p, rule=CON6_rule)


# CON7 - SPARSE
def CON7_rule(model, p):
    y2_terms = [model.Y2[idx] for idx in model.Y2_index if idx[0] == p]
    if len(y2_terms) == 0:
        return Constraint.Infeasible
    return sum(y2_terms) == 1

model.CON7 = Constraint(model.p, rule=CON7_rule)


# Demand satisfaction - SPARSE
def DEM_rule(model):
    if len(model.INH_index) == 0:
        return Constraint.Feasible
    return sum(model.INH[idx] for idx in model.INH_index) <= len(model.p)

model.DEM = Constraint(rule=DEM_rule)


# Flow constraints - SPARSE
def CON8_rule(model, p, c, m, j, t):
    if (p, c, m, j, t) not in model.Y1_index:
        return Constraint.Skip
    return model.LSR[p, c, m, j, t] >= model.Y1[p, c, m, j, t] * model.FMIN

model.CON8 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON8_rule)


def CON9_rule(model, p, c, m, j, t):
    if (p, c, m, j, t) not in model.Y1_index:
        return Constraint.Skip
    return model.LSR[p, c, m, j, t] <= model.Y1[p, c, m, j, t] * model.FMAX

model.CON9 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON9_rule)

def CON10_rule(model, p, m, h, j, t):
    if (p, m, h, j, t) not in model.Y2_index:
        return Constraint.Skip
    return model.MSO[p, m, h, j, t] >= model.Y2[p, m, h, j, t] * model.FMIN

model.CON10 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON10_rule)


def CON11_rule(model, p, m, h, j, t):
    if (p, m, h, j, t) not in model.Y2_index:
        return Constraint.Skip
    return model.MSO[p, m, h, j, t] <= model.Y2[p, m, h, j, t] * model.FMAX

model.CON11 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON11_rule)


# CON12-15 - SPARSE with proper handling
def CON12_rule(model, p):
    lhs_terms = [model.Y2[idx] for idx in model.Y2_index if idx[0] == p and idx[2] == 'h1']
    rhs = sum(value(model.INC[p, 'c1', t]) for t in model.t)
    if len(lhs_terms) == 0:
        return Constraint.Feasible if rhs == 0 else Constraint.Infeasible
    return sum(lhs_terms) == rhs

model.CON12 = Constraint(model.p, rule=CON12_rule)


def CON13_rule(model, p):
    lhs_terms = [model.Y2[idx] for idx in model.Y2_index if idx[0] == p and idx[2] == 'h2']
    rhs = sum(value(model.INC[p, 'c2', t]) for t in model.t)
    if len(lhs_terms) == 0:
        return Constraint.Feasible if rhs == 0 else Constraint.Infeasible
    return sum(lhs_terms) == rhs

model.CON13 = Constraint(model.p, rule=CON13_rule)


def CON14_rule(model, p):
    lhs_terms = [model.Y2[idx] for idx in model.Y2_index if idx[0] == p and idx[2] == 'h3']
    rhs = sum(value(model.INC[p, 'c3', t]) for t in model.t)
    if len(lhs_terms) == 0:
        return Constraint.Feasible if rhs == 0 else Constraint.Infeasible
    return sum(lhs_terms) == rhs

model.CON14 = Constraint(model.p, rule=CON14_rule)


def CON15_rule(model, p):
    lhs_terms = [model.Y2[idx] for idx in model.Y2_index if idx[0] == p and idx[2] == 'h4']
    rhs = sum(value(model.INC[p, 'c4', t]) for t in model.t)
    if len(lhs_terms) == 0:
        return Constraint.Feasible if rhs == 0 else Constraint.Infeasible
    return sum(lhs_terms) == rhs

model.CON15 = Constraint(model.p, rule=CON15_rule)


# ============================================================================
# TIME CONSTRAINTS
# ============================================================================

def TCON_rule(model, p):
    return model.TRT[p] <= model.ND

model.TCON = Constraint(model.p, rule=TCON_rule)


def START_rule(model, p):
    return model.STT[p] == sum(model.INC[p, c, t] * t for c in model.c for t in model.t)

model.START = Constraint(model.p, rule=START_rule)


def END_rule(model, p):
    inh_terms = [(model.INH[idx], idx[2]) for idx in model.INH_index if idx[0] == p]
    if len(inh_terms) == 0:
        return model.CTT[p] == 0
    return model.CTT[p] == sum(var * t for var, t in inh_terms)

model.END = Constraint(model.p, rule=END_rule)


def TSEQ_rule(model, p):
    return model.STT[p] <= model.CTT[p]

model.TSEQ = Constraint(model.p, rule=TSEQ_rule)


def TIME_rule(model, p):
    return model.TRT[p] == model.CTT[p] - model.STT[p]

model.TIME = Constraint(model.p, rule=TIME_rule)


def ATIME_rule(model):
    return model.ATRT == sum(model.TRT[p] for p in model.p) / len(model.p)

model.ATIME = Constraint(rule=ATIME_rule)


# =======================================================================================
# BUILD AND SOLVE
# =======================================================================================
timer = TicTocTimer()
timer.tic('start')
print('=' * 100)
print('SPARSE VARIABLE VERSION - WITH QUEUE')
print('=' * 100)
print('-----------------------------------------------Building model-----------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------')
instance = model.create_instance(data)
timer.toc('Built model')

# Print variable counts
print(f"\nVariable counts:")
print(f"  Y1: {len(instance.Y1)} (was {len(instance.p)*len(instance.c)*len(instance.m)*len(instance.j)*len(instance.t)})")
print(f"  Y2: {len(instance.Y2)} (was {len(instance.p)*len(instance.m)*len(instance.h)*len(instance.j)*len(instance.t)})")
print(f"  Reduction: {100*(1 - len(instance.Y1)/(len(instance.p)*len(instance.c)*len(instance.m)*len(instance.j)*len(instance.t))):.1f}%")


print('-----------------------------------------------Solving model------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------')
opt = SolverFactory('gurobi')
myoptions = {
    'OutputFlag': 1,
    'MIPGap': 0,
    'Presolve': 2,
    'Method': 2,
    'Cuts': 2,
    'Threads': 0,
    'TimeLimit': 7200,
}

results = opt.solve(instance, options=myoptions, tee=True)
timer.toc('Time to solve')
t1_stop = process_time()


print('------------------------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------RESULTS---------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------RATIO(m,t)-------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for t in instance.t:
    for m in instance.m:
        if value(instance.RATIO[m,t])*100 > 1e-3:
            print('The percentage of utilisation of MS site', m, 'at time','t{}'.format(t), 'is', value(instance.RATIO[m,t])*100,'%')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')

# NEW: Print queue times
print('------------------------------------------------QTIME(p)---------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    if value(instance.QTIME[p]) > 0.001:
        print('Queue time for therapy', p, 'is', value(instance.QTIME[p]), 'days')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------OUTC(p,c,t)------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for c in instance.c:
        for t in instance.t:
            if (p, c, t) in instance.OUTC_index and value(instance.OUTC[p,c,t]) == 1:
                print('Therapy',p,'leaving leukapheresis site',c,'at time','t{}'.format(t))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------LSR(p,c,m,j,t)---------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for idx in instance.LSR_index:
    if value(instance.LSR[idx]) == 1:
        print('Therapy',idx[0],'leaving LS',idx[1], 'arriving at MS',idx[2],'with transport mode',idx[3],'at time','t{}'.format(idx[4]))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------INM(p,m,t)-------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for m in instance.m:
        for t in instance.t:
            if value(instance.INM[p,m,t]) == 1:
                print('Therapy',p,'entering manufacturing site',m,'at time','t{}'.format(t))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')

# NEW: Print manufacturing start times
print('------------------------------------------------STARTM(p,m,t)----------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for m in instance.m:
        for t in instance.t:
            if value(instance.STARTM[p,m,t]) == 1:
                print('Therapy',p,'starts manufacturing at site',m,'at time','t{}'.format(t))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')

print('------------------------------------------------DURV(p,m,t)-------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for m in instance.m:
        for t in instance.t:
            if t<130:
                if (value(instance.DURV[p,m,t+1])-value(instance.DURV[p,m,t]))**2 == 1:
                    print('Therapy',p,'is manufactured at site',m,'for the time period','t{}'.format(t),'-','t{}'.format(t+ instance.TMFE))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------OUTM(p,m,t)------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    for m in instance.m:
        for t in instance.t:
            if value(instance.OUTM[p,m,t]) == 1:
                print('Therapy',p,'leaving manufacturing site', m,'at time','t{}'.format(t))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')                
print('------------------------------------------------LSA(p,c,m,j,t)---------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for idx in instance.LSA_index:
    if value(instance.LSA[idx]) == 1:
        print('Therapy',idx[0],'that left LS', idx[1],'arriving at MS',idx[2],'with transport mode', idx[3],'at time','t{}'.format(idx[4]))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------MSO(p,m,h,j,t)----------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for idx in instance.MSO_index:
    if value(instance.MSO[idx]) == 1:
        print('Therapy',idx[0],'leaving MS',idx[1], 'arriving at hospital',idx[2], 'with transport mode',idx[3],'at time','t{}'.format(idx[4]))
                        
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')                 
print('------------------------------------------------FTD(p,m,h,j,t)----------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for idx in instance.FTD_index:
    if value(instance.FTD[idx]) == 1:
        print('Final therapy',idx[0],'that left MS',idx[1],'arriving at hospital',idx[2], 'with transport mode',idx[3],'at time','t{}'.format(idx[4]))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')                        
print('------------------------------------------------INH(p,h,t)-------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
for idx in instance.INH_index:
    if value(instance.INH[idx]) == 1:
        print('Therapy',idx[0],'arriving at hospital',idx[1],'at time','t{}'.format(idx[2]))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------CTM(p)------------------------------------------------------------------') 
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    print('Total manufacturing cost','of therapy' ,p,'is',value(instance.CTM[p]))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')    
print('------------------------------------------------TTC(p)------------------------------------------------------------------') 
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    print('Total transport cost','of therapy' ,p,'is',value(instance.TTC[p]))

print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')    
print('------------------------------------------------TRT(p)------------------------------------------------------------------') 
print('-----------------------------------------------------------------------------------------------------------------------')
for p in instance.p:
    print('Total return time','of therapy' ,p,'is',value(instance.TRT[p]))
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('Manufacturing facilities to be established')
for m in instance.m:
    if value(instance.E1[m])>0.5:  
        print(m)
        
obj_val = value(instance.obj)
print('Total cost=',obj_val)
print('Average manufacturing cost per therapy', value(sum(instance.CTM[p] for p in instance.p))/len(instance.p))
print('Average transport cost per therapy', value(sum(instance.TTC[p] for p in instance.p))/len(instance.p))
print('Average QC cost per therapy',(10476+9312))
print('Average cost per therapy',obj_val/len(instance.p))
print('Average return time =',np.rint(value(instance.ATRT)))
print('Average queue time =', sum(value(instance.QTIME[p]) for p in instance.p)/len(instance.p), 'days')
print("CPU time:", t1_stop-t1_start)
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------------')




fig = plt.figure(figsize=(10,10))
Nc=4
Nm=6
Nh=4
leukapheresisY=np.linspace(0.1,0.9,Nc)
leukapheresisX=0.1*np.ones_like(leukapheresisY)
manufacturingY=np.linspace(0.1,0.9,Nm)
manufacturingX=0.5*np.ones_like(manufacturingY)
hospitalY=np.linspace(0.1,0.9,Nh)
hospitalX=0.9*np.ones_like(hospitalY)
for c in instance.c:
    for m in instance.m:
        if (value(instance.X1[c,m]))!=0:
            plt.plot([leukapheresisX[Nc-instance.c.ord(c)],manufacturingX[Nm-instance.m.ord(m)]],[leukapheresisY[Nc-instance.c.ord(c)],manufacturingY[Nm-instance.m.ord(m)]],color='darkmagenta')

for m in instance.m:
    for h in instance.h:
        if (value(instance.X2[m,h]))!=0:
            plt.plot([manufacturingX[Nm-instance.m.ord(m)],hospitalX[Nh-instance.h.ord(h)]],[manufacturingY[Nm-instance.m.ord(m)],hospitalY[Nh-instance.h.ord(h)]],color='darkmagenta')
            

for i in range(0,Nc):
    plt.scatter(leukapheresisX[i],leukapheresisY[i],s=250,color='teal')
    plt.text(leukapheresisX[Nc-i-1]-0.05,leukapheresisY[Nc-i-1],'C'+str(i+1),fontweight='bold')
for i in range(0,Nm):
    plt.scatter(manufacturingX[i],manufacturingY[i],s=250,color='teal')
    plt.text(manufacturingX[Nm-i-1]+0.03,manufacturingY[Nm-i-1],'M'+str(i+1),fontweight='bold')
for i in range(0,Nh):
    plt.scatter(hospitalX[i],hospitalY[i],s=250,color='teal')
    plt.text(hospitalX[Nh-i-1]+0.03,hospitalY[Nh-i-1],'H'+str(i+1),fontweight='bold')    
plt.axis('off')

fig.savefig('network.png')
