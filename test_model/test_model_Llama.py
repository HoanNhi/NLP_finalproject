import fitz  #import PyMuPDF module
import os #module used to interact with local Operating System.
import pytesseract
import matplotlib.pyplot as plt
import cv2
os.environ['HF_HOME'] = '/mnt/data/thomas/.cache' #Choose where to save llama
import torch
import transformers


prompts_1 = []
system_role = {"role": "system", "content": "You are a helpful academic assistant trained to answer questions about academic policy concisely."}
prompts_1.append({"role": "user", "content": "Does withdrawal from a course after 4th week result in a 4 credits deduction in the total credits covered by financial aid?"})
prompts_1.append({"role": "user", "content": "What are the consequences if I withdraw from capstone II?"})
prompts_1.append({"role": "user", "content": "Can I continue my study for more than 4 years with financial aid?"})
prompts_1.append({"role": "user", "content": "Can you differentiate a cross-listed course and elective applied course in the case of a double major?"})
prompts_1.append({"role": "user", "content": "I would like to ask for the capstone withdrawal policy. What will be the impact when I choose to drop the capstone before the Fall 2025 term starts? Will there be any penalties associated with withdrawing, such as a 'W' notation on my transcript?"})
prompts_1.append({"role": "user", "content": "I have taken Microeconomic Analysis and Macroeconomic Analysis in Fall 2023. Am I eligible to take Financial Economics in Spring 2024?"})

prompts_2 = []
prompts_2.append({'role': 'user', 'content': 'Context: There are earned credits and attempted credits. Earned credits refer to the number of credit hours a student successfully completes with a grade of D or higher for letter grade options and a Pass for Pass/No Pass options. Attempted credits are the credit hours students enroll in and are recorded in the transcript, regardless of whether they complete the courses or pass them. This includes all courses for which a student receives a grade, including withdrawals (W), incompletes (I), repeats (RP), and failures (F). See section 4 for more details on different types of grades. Question: Does withdrawal from a course after 4th week result in a 4 credits deduction in the total credits covered by financial aid?'})
prompts_2.append({'role': 'user', 'content': 'Context: All majors at Fulbright offer an optional capstone pathway to be completed during the student’s final year. Students may apply for the opportunity to fulfill 8 credits of their major requirements via a capstone project. A student wishing to do a capstone must apply during the term prior to the first term of the capstone (Capstone I), and may only apply after having completed 80 credits of coursework. Question: What are the consequences if I withdraw from capstone II?'})
prompts_2.append({'role': 'user', 'content': 'Context: Fulbright follows MOET regulations, which state that the maximum time to complete the program cannot exceed double the standard duration. Question: Can I continue my study for more than 4 years with financial aid?'})
prompts_2.append({'role': 'user', 'content': 'Context: For Applied Mathematics, Elective Applied Courses (2 required). Students pursuing a double major must complete all requirements for each major. If two majors (or two minors, or a major and a minor) share cross- listed or tagged1 courses, up to 8 credits (two courses) can be used to satisfy both sets of requirements. Question: Can you differentiate a cross-listed course and elective applied course in the case of a double major?'})
prompts_2.append({'role': 'user', 'content': """Context: All majors at Fulbright offer an optional capstone pathway to be completed during the student’s final year. Students may apply for the opportunity to fulfill 8 credits of their major requirements via a capstone project. A student wishing to do a capstone must apply during the term prior to the first term of the capstone (Capstone I), and may only apply after having completed 80 credits of coursework. Students who do not do a capstone will need to fulfill 8 credits of coursework as specified in their major. Individual majors will determine eligibility requirements, and students should discuss their proposals with program coordinators and potential advisors to prepare their applications. Capstone projects will be graded No Pass/Pass/Honors and must be evaluated by the supervising faculty at least one month before the end of the student’s graduating term. Students whose capstone is graded Honors will be awarded Graduation with Honors, answer this question: I would like to ask for the capstone withdrawal policy. What will be the impact when I choose to drop the capstone before the Fall 2025 term starts? Will there be any penalties associated with withdrawing, such as a 'W' notation on my transcript?"""})
prompts_2.append({'role': 'user', 'content': 'Context: Econometrics: This course is concerned with the application of statistical theory to the analysis of economic data and the estimation of economic relationships. The course focuses on regression analysis and its uses in empirical economic research. Students will learn how to construct economic models and test them with data. Microeconomic Analysis: This course focuses on how incentives both constrain and direct the decision making of consumers, producers, and governments. Students will learn to use both graphical and optimization techniques to solve the problems faced by consumers (what to buy), producers (what to produce and what price to sell it at), and governments (which policies to enact). Macroeconomic Analysis: In this course, students will combine empirical observations and economic models to study the dynamics of the aggregate economy. This course focuses on the macroeconomic tools of government – fiscal and monetary policy – and their effects on long-run economic growth, employment, and inflation. Question: I have taken Microeconomic Analysis and Macroeconomic Analysis in Fall 2023. Am I eligible to take Financial Economics in Spring 2024?'})

prompts_3 = []
prompts_3.append({'role': 'user', 'content': 'Context: There are earned credits and attempted credits. Earned credits refer to the number of credit hours a student successfully completes with a grade of D or higher for letter grade options and a Pass for Pass/No Pass options. Attempted credits are the credit hours students enroll in and are recorded in the transcript, regardless of whether they complete the courses or pass them. This includes all courses for which a student receives a grade, including withdrawals (W), incompletes (I), repeats (RP), and failures (F). See section 4 for more details on different types of grades. There are two Withdrawal periods per term. The first period starts after 4:00 PM Friday, Week 2 to 4:00 PM Friday, Week 4. During this time, students can request to withdraw from the course without any consequence. The second period starts after 4:00 PM Friday, Week 4 to 4:00 PM Friday, Week 8. Attempted credits and a "W" grade will be recorded on the transcript for the course(s) from which students withdraw in this period. This grade does not affect a student’s GPA. In order to withdraw from a course, students need to submit the Course Withdrawal form to the Registrar’s Office, following the above timeline. A student who withdraws from a course beyond the halfway point of that course (end of week 4 for an 8-week course or end of week 8 for a full-term course) automatically receives a grade of ‘F’, which will affect the student’s GPA. Exceptions will only be permitted in exceptional circumstances (see Section 12), and it requires supporting documentation as well as the authorization of course instructor(s), the student’s academic/major advisor, and the Registrar’s Office. Question: Does withdrawal from a course after 4th week result in a 4 credits deduction in the total credits covered by financial aid?'})

prompts_3.append({'role': 'user', 'content': 'Context: All majors at Fulbright offer an optional capstone pathway to be completed during the student’s final year. Students may apply for the opportunity to fulfill 8 credits of their major requirements via a capstone project. A student wishing to do a capstone must apply during the term prior to the first term of the capstone (Capstone I), and may only apply after having completed 80 credits of coursework. Students who do not do a capstone will need to fulfill 8 credits of coursework as specified in their major. Individual majors will determine eligibility requirements, and students should discuss their proposals with program coordinators and potential advisors to prepare their applications. Capstone projects will be graded No Pass/Pass/Honors and must be evaluated by the supervising faculty at least one month before the end of the student’s graduating term. Students whose capstone is graded Honors will be awarded Graduation with Honors. Question: What are the consequences if I withdraw from capstone II?'})

prompts_3.append({'role': 'user', 'content': 'Context: Fulbright follows MOET regulations, which state that the maximum time to complete the program cannot exceed double the standard duration. In other words, students have a maximum of 8 years to complete their studies. Question: Can I continue my study for more than 4 years with financial aid?'})

prompts_3.append({'role': 'user', 'content': 'Context: For Applied Mathematics, These are the courses from other majors that emphasize the applications of Mathematics.  Students will have to take at least 2 applied courses (at least one at 200 level). If there is any prerequisite, students need to either complete the course’s prerequisites or obtain the instructor’s approval. Students pursuing a double major must complete all requirements for each major. If two majors (or two minors, or a major and a minor) share cross-listed or tagged courses, up to 8 credits (two courses) can be used to satisfy both sets of requirements. This limit is per each pair of majors (or major and minor) and does not restrict the total number of double-counted credits or courses a student can claim. For example, a student can major in both A and B and minor in C with a total of 6 double-counted courses: two in A and B, two in B and C, and two in A and C. This rule does not apply to courses that are not cross-listed or tagged. For instance, if major A requires certain courses from major B for out-of-area foundation, exploration, or applications of the major, and these courses are not cross-listed between A and B, they do not count toward the 8-credit limit. Question: Can you differentiate a cross-listed course and elective applied course in the case of a double major?'})


prompts_3.append({'role': 'user', 'content': """Context: All majors at Fulbright offer an optional capstone pathway to be completed during the student’s final year. Students may apply for the opportunity to fulfill 8 credits of their major requirements via a capstone project. A student wishing to do a capstone must apply during the term prior to the first term of the capstone (Capstone I), and may only apply after having completed 80 credits of coursework. Students who do not do a capstone will need to fulfill 8 credits of coursework as specified in their major. Individual majors will determine eligibility requirements, and students should discuss their proposals with program coordinators and potential advisors to prepare their applications. Capstone projects will be graded No Pass/Pass/Honors and must be evaluated by the supervising faculty at least one month before the end of the student’s graduating term. Students whose capstone is graded Honors will be awarded Graduation with Honors. Attempted credits and a "W" grade will be recorded on the transcript for the course(s) from which students withdraw in this period. This grade does not affect a student’s GPA. In order to withdraw from a course, students need to submit the Course Withdrawal form to the Registrar’s Office, following the above timeline. A student who withdraws from a course beyond the halfway point of that course (end of week 4 for an 8-week course or end of week 8 for a full-term course) automatically receives a grade of ‘F’, which will affect the student’s GPA. Question: I would like to ask for the capstone withdrawal policy. What will be the impact when I choose to drop the capstone before the Fall 2025 term starts? Will there be any penalties associated with withdrawing, such as a 'W' notation on my transcript?"""})

prompts_3.append({'role': 'user', 'content': 'Context: Econometrics: This course is concerned with the application of statistical theory to the analysis of economic data and the estimation of economic relationships. The course focuses on regression analysis and its uses in empirical economic research. Students will learn how to construct economic models and test them with data. Microeconomic Analysis: This course focuses on how incentives both constrain and direct the decision making of consumers, producers, and governments. Students will learn to use both graphical and optimization techniques to solve the problems faced by consumers (what to buy), producers (what to produce and what price to sell it at), and governments (which policies to enact). Macroeconomic Analysis: In this course, students will combine empirical observations and economic models to study the dynamics of the aggregate economy. This course focuses on the macroeconomic tools of government – fiscal and monetary policy – and their effects on long-run economic growth, employment, and inflation. Advanced level courses in Economics can be taken after Microeconomic and Macroeconomic Analysis and Econometrics. Financial Economics is counted as an advanced course. Question: I have taken Microeconomic Analysis and Macroeconomic Analysis in Fall 2023. Am I eligible to take Financial Economics in Spring 2024?'})

model_llama_33_70B = "meta-llama/Llama-3.3-70B-Instruct"
pipeline_llama_33_70B = transformers.pipeline(
"text-generation",
model=model_llama_33_70B,
model_kwargs={"torch_dtype": torch.bfloat16},
device_map="auto",
)

print("-"*30 + "Starting question 1" +"-"*30)
for question in prompts_1:
    questions = []
    questions.append(system_role)
    questions.append(question)
    outputs = pipeline_llama_33_70B(questions, max_new_tokens = 1024, num_beams = 2)
    print("The answer by Llama 3.3 70B is: ")
    print(outputs[0]['generated_text'][-1])
    print("-"*100)

print("-" * 30 + "Starting question 2" + "-" * 30)
# Qwene 7B-1M
for question in prompts_2:
    questions = []
    questions.append(system_role)
    questions.append(question)
    outputs = pipeline_llama_33_70B(questions, max_new_tokens=1024, num_beams=2)
    print("The answer by Llama 3.3 70B is: ")
    print(outputs[0]['generated_text'][-1])
    print("-" * 100)

print("-" * 30 + "Starting question 3" + "-" * 30)
for question in prompts_2:
    questions = []
    questions.append(system_role)
    questions.append(question)
    outputs = pipeline_llama_33_70B(questions, max_new_tokens = 1024, num_beams = 2)
    print("The answer by Llama 3.3 70B is: ")
    print(outputs[0]['generated_text'][-1])
    print("-"*100)