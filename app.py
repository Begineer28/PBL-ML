import streamlit as st
import joblib

def main():
    html_temp="""
    <div style="background-color:lightblue;padding:16px">
    <h2 style="color:black";text-align:center> Heart disease predication using ML </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    model=joblib.load('joblib_model')
    
    p1=st.slider("Enter your age",18,100)
    
    s1=st.selectbox("Sex",("MALE","FEMALE"))
    
    if s1=="MALE":
        P2=1
    elif s1=="FEMALE":
        p2=0
        
    p3=st.slider("Enter chest pain",0,3)
    
    p4=st.number_input("Enter resting blood pressure(trestbps)-")
    p5=st.number_input("Enter serum cholestero(mg/dL)-")
    
    s2=st.selectbox('Do you have fasting blood sugar(fbs)?',("Yes","No"))
    
    if s2=="Yes":
        P6=1
    elif s2=="No":
        p6=0
        
    s3=st.selectbox('Do you have resting electrocardiographic results(restecg)?',("Yes","No"))
     
    if s3=="Yes":
         P7=1
    elif s3=="No":
         p7=0
    
    p8=st.number_input("Enter maximum heart rate achieved (thalach)-")
                       
    s4=st.selectbox('Do you have exercise induced angina(exang)?',("Yes","No"))
    if s4=="Yes":
         P9=1
    elif s4=="No":
         p9=0
         
    p10=st.number_input("Enter oldpeak(0-5)-")
    
    p11=st.slider("Enter slope of elevation",0,2)
    p12=st.slider("Enter number of major vessels colored by fluoroscopy(ca)",0,4)
    
    p13=st.number_input("Enter stage of thalassemia (thal[1-3])-")
   
    if st.button("predict") :
        pred=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
        
        st.success("Probablity of heart disease is-{}".format(pred[0]))       
        
       
        
if __name__=='__main__':
    main()