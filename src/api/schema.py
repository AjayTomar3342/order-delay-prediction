from pydantic import BaseModel


class OrderInput(BaseModel):
    # -----------------------------
    # Numerical Features
    # -----------------------------
    Days_for_shipment_scheduled: float
    order_hour: float

    # -----------------------------
    # Categorical Features
    # -----------------------------
    Type: str
    Category_Name: str
    Customer_City: str
    Customer_Country: str
    Customer_Segment: str
    Customer_State: str
    Department_Name: str
    Market: str
    Order_City: str
    Order_Country: str
    Order_Region: str
    Order_State: str
    Shipping_Mode: str

    class Config:
        populate_by_name = True
