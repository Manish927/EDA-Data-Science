from pydantic import BaseModel


class BackorderRequest(BaseModel):

    national_inv: int
    lead_time: float
    in_transit_qty: int
    forecast_3_month: int
    forecast_6_month: int
    forecast_9_month: int
    sales_1_month: int
    sales_3_month: int
    sales_6_month: int
    sales_9_month: int
    min_bank: int
    pieces_past_due: int
    perf_6_month_avg: float
    perf_12_month_avg: float
    local_bo_qty: int