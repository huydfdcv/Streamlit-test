import streamlit as st

def sum(a,b):
    return a+b

def main():
    st.title("tinh tong 2 so")
    a = st.number_input("nhap a")
    b = st.number_input("nhap b")
    
    if st.button("Tinh tong"):
        result = sum(a,b)
        st.write(f"Tong 2 so la {result}")

main()