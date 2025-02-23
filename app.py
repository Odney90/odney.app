'proba_Nul': proba_Nul,  
        })  


        # Generation of the DOC report  
        if st.button("📄 Download the Report"):  
            doc = generate_report(st.session_state.history)  
            buffer = io.BytesIO()  
            doc.save(buffer)  
            buffer.seek(0)  
            st.download_button("Download the report", buffer, "predictions_report.docx")  


        # Summary table of results  
        st.subheader("📊 Summary Table of Results")  
        data = {  
            "Team": ["Team A", "Team B", "Draw"],  
            "Predicted Probability": [f"{proba_A:.2%}", f"{proba_B:.2%}", f"{proba_Nul:.2%}"],  
            "Predicted Odds": [f"{predicted_odds_A:.2f}", f"{predicted_odds_B:.2f}", f"{predicted_odds_Nul:.2f}"],  
            "Bookmaker Odds": [  
                f"{st.session_state.data['bookmaker_odds_A']:.2f}",  
                f"{st.session_state.data['bookmaker_odds_B']:.2f}",  
                f"{st.session_state.data['bookmaker_odds_Nul']:.2f}",  
            ],  
            "Value Bet": [  
                "✅" if predicted_odds_A < st.session_state.data['bookmaker_odds_A'] else "❌",  
                "✅" if predicted_odds_B < st.session_state.data['bookmaker_odds_B'] else "❌",  
                "✅" if predicted_odds_Nul < st.session_state.data['bookmaker_odds_Nul'] else "❌",  
            ],  
        }  
        df_results = pd.DataFrame(data)  
        st.table(df_results)  


        # Reminder message about Value Bet  
        st.markdown("""  
        ### 💡 What is a Value Bet?  
        A **Value Bet** is a bet where the predicted odds by the model are **lower** than the odds offered by the bookmaker.   
        This indicates that the bookmaker underestimates the probability of this event, making it a potentially profitable opportunity.  
        """)  


    # Tab to display the weights of the criteria  
    with tab2:  
    st.subheader("📊 Weights of the Random Forest Model Criteria")  
    if st.session_state.criteria_weights:  
        weights_df = pd.DataFrame({  
            'Criteria': [  
                'Score Rating A', 'Goals Scored A', 'Goals Conceded A', 'Average Possession A',  
                'Expected Goals A', 'Expected Goals Against A', 'Shots on Target A', 'Big Chances A',  
                'Absences A', 'Recent Form A', 'Score Rating B', 'Goals Scored B', 'Goals Conceded B',  
                'Average Possession B', 'Expected Goals B', 'Expected Goals Against B', 'Shots on Target B',  
                'Big Chances B', 'Absences B', 'Recent Form B'  
            ],  
            'Weight': st.session_state.criteria_weights  
        })
                # Display weights of criteria in a table  
        st.table(weights_df)  


        # Visualization of criteria weights with Plotly  
        fig = px.bar(weights_df, x='Criteria', y='Weight', title='Weights of the Random Forest Model Criteria',   
                      labels={'Weight': 'Weight', 'Criteria': 'Criteria'}, color='Weight')  
        st.plotly_chart(fig)  


        # Visualization of criteria weights with Altair  
        alt_chart = alt.Chart(weights_df).mark_bar().encode(  
            x=alt.X('Criteria:N', sort='-y'),  
            y='Weight:Q',  
            color='Weight:Q'  
        ).properties(  
            title='Weights of the Random Forest Model Criteria'  
        )  
        st.altair_chart(alt_chart, use_container_width=True)  


# Section for head-to-head between teams  
st.markdown("### 🤝 Head-to-Head between the Teams")
