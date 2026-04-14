from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import math

app = FastAPI(title="超声波声速测量计算API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExperimentData(BaseModel):
    f_kHz: float
    t_initial: float
    t_final: float
    resonance_xi: List[float]
    phase_xi: List[float]

DELTA_X_INSTR = 0.02  # mm

def theoretical_speed(t_celsius: float) -> float:
    u0 = 331.45
    return u0 * math.sqrt(1 + t_celsius / 273.15)

def process_method(xi: List[float], method: str, f_hz: float, t_avg: float = 25.0) -> Dict[str, Any]:
    """
    method: 'resonance' 或 'phase'
    返回包含所有计算中间值和最终结果的字典
    """
    if len(xi) != 12:
        raise ValueError("需要12个位置数据")
    
    # 1. 计算 L_i
    L = [abs(xi[i+6] - xi[i]) for i in range(6)]
    L_mean = np.mean(L)
    
    # 2. 不确定度
    u_A = np.std(L, ddof=1)
    u_B = math.sqrt(2) * DELTA_X_INSTR
    u_L = math.sqrt(u_A**2 + u_B**2)
    
    # 3. 波长及其不确定度
    if method == 'resonance':
        wavelength = 2 * L_mean / 6          # λ = 2L̄/6
        u_wavelength = u_L / 3
    else:  # phase
        wavelength = L_mean / 6              # λ = L̄/6
        u_wavelength = u_L / 6
    
    # 4. 声速
    u_sound = f_hz * wavelength / 1000       # 波长 mm → m
    u_f = f_hz * 0.002 / 100                # 0.002%
    
    # 5. 相对不确定度
    rel_u_wavelength = u_wavelength / wavelength if wavelength != 0 else 0
    rel_u_f = u_f / f_hz if f_hz != 0 else 0
    E_u = math.sqrt(rel_u_wavelength**2 + rel_u_f**2)
    U_u = u_sound * E_u
    
    # 6. 百分误差（需要理论值，暂时用传入的 t_avg 计算，外部会覆盖）
    u_theory = theoretical_speed(t_avg)
    percent_error = abs((u_sound - u_theory) / u_theory) * 100 if u_theory != 0 else 0
    
    # 7. 生成详细的代入数字过程（供前端显示）
    # 这些字符串只是辅助，前端也可以自己生成，但为了保持一致，后端提供基础数据即可
    # 注意：前端会自己生成“三步走”文本，因此后端只需要提供原始数值，不需要提供 formula_details
    # 但为了兼容之前的调用，我们仍然返回必要的字段
    return {
        "L_values": L,
        "L_mean": L_mean,
        "u_A": u_A,
        "u_B": u_B,
        "u_L": u_L,
        "wavelength_mm": wavelength,
        "u_wavelength": u_wavelength,
        "u_sound": u_sound,
        "u_f": u_f,
        "E_u": E_u,
        "U_u": U_u,
        "percent_error": percent_error
    }

@app.post("/calculate")
async def calculate(data: ExperimentData):
    try:
        # 计算平均温度
        t_avg = (data.t_initial + data.t_final) / 2.0
        u_theory = theoretical_speed(t_avg)
        f_hz = data.f_kHz * 1000.0
        
        # 处理两种方法
        res = process_method(data.resonance_xi, "resonance", f_hz, t_avg)
        phase = process_method(data.phase_xi, "phase", f_hz, t_avg)
        
        # 重新计算百分误差（使用实际理论值，避免占位温度）
        res["percent_error"] = abs((res["u_sound"] - u_theory) / u_theory) * 100
        phase["percent_error"] = abs((phase["u_sound"] - u_theory) / u_theory) * 100
        
        return {
            "success": True,
            "temperature": {
                "t_initial": data.t_initial,
                "t_final": data.t_final,
                "t_average": round(t_avg, 2),
                "u_theory": round(u_theory, 4)
            },
            "resonance": res,
            "phase": phase
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
