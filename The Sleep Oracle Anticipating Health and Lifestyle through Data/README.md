# EDA-Report-on-Sleep-Health-LifeStyle-DataSet :
1.Using the dataset of sleep health and lifestyle,aims to determine a machine learning model that can accurately predict the sleep disorder of a person and determine which attributes/features of a person affects the sleep disorder classification by using feature importance.

 ![3](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/dbb4eb81-c33c-4d41-a35d-92d3e7c645e1)
   
2. The exploratory data analysis reveals several interesting insights into the relationship between sleep health and lifestyle factors. Younger individuals tend to sleep longer, and daily exercise is associated with slightly longer sleep durations. However, higher stress levels,BMI ,Physical Activity Level are associated with poorer sleep quality. These findings highlight the importance of adopting a healthy lifestyle to promote better sleep and overall well-being. Further analysis, such as regression models, could be conducted to explore the relative impact of lifestyle factors on sleep health more comprehensively.
   
 ![1](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/0bf293a7-5805-4515-98fc-9f63770b052f)

   
4.## EDA is performed by using following Python Libraries : 
- Numpy
- Pandas
- Matplotlib
- Seaborn
  
5. ## Following are the snapshots of different charts generated in this EDA process :

 ### GENDER value Representation by Countplot :
  ![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/6d876739-a759-40a8-a6b5-203132e37574)

### GENDER value Representation by Countplot Using Containers:
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/d380b1ac-372e-44d8-897d-d7f262a8b4f8)

###  Values containing bar graph Represented  by countplot:
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/2743f18f-70ef-4fd1-b51a-08e403b6a12d)

### Profession value Representation by Countplot :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/a1a494cf-2137-4d7e-b7d2-71ff28e46c6a)

### Representation of histplot relation between Sleep Duration and count:
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/9cd90d75-1685-4281-8c64-72070b82cb04)

### Visualizing the distribution of the Quality of Sleep:
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/6bb22ee1-39b3-4c8c-80ef-d686c0bee139)

### Visualizing relationship between Quality of sleep and Sleep Duration Comparing data across groups :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/3b92f5af-020c-4b5d-8ac1-8f0ffc9f49dd)

### Exploring the relationship between sleep duration and quality of sleep :
!3[image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/abf00722-b368-483f-b9aa-ccd1977665ab)

### Visualizing Distplot between age & count :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/2023aa5a-30dd-4167-91f5-0c47ec944038)

### Visualizing Boxplot of Age:
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/9694a503-703b-459e-ab33-74ae4a00ae7d)

### Visualizing Boxplot on Heart Rate :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/58aa8d9c-b758-4c1d-b594-c36bb320afbf)

### Visualising Boxplot On Stress Level :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/ee354369-bdd2-45ba-8c13-a75b393d5a60)

### Visualizing Distplot For Heart Rate :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/a87f5650-aea4-42fe-81a3-40a238081062)

### Pair- plot Distribution :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/4d160d33-5082-43ac-b422-ff446c784014)

### Representation of Correlations Using Heatmap :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/f8c0fb9e-34fb-4a96-890c-7b65ce051472)

### Exploring the relationship between physical activity level and quality of sleep using Bargraph:
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/019d11ec-f067-494e-b087-11545e4c83ca)

### Exploring the relationship between BMI category and quality of sleep :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/64d1d276-e4c1-437d-8168-044820b9c495)

### Exploring the relationship between stress level and quality of sleep :
![image](https://github.com/sakshipaidalwar/EDA-Report-on-Sleep-Health-LifeStyle-DataSet/assets/105778333/7da5c7cf-4ebb-4f05-9fef-835dcde7f90b)

## Obervations
- The "Age" column in the dataset . It contains numerical values that range from 27 to 59.
- The average age (Mean) of the people in the dataset, which is approximately 42.18 years.
- The standard deviation is approximately 8.67 years minimum age value is 27 .
- It means that 25% of the ages are below or equal to 35.25 years.
- The second quartile  represents the middle value of the ages when they are arranged in ascending order. In this case, the median is 43 years, which means that 
  50% of the ages are below or equal to 43. 75% .
- This is the third quartile or the 75th percentile. It means that 75% of the ages are below or equal to 50 years. max: This is the maximum value in the "Age" 
  column, which is 59. It represents the oldest age in the dataset
- The box plot reveals the variation in age across different Sleep Disorder categories. There are three Sleep Disorder categories represented: "sleep apnea," 
  "Insomnia," and "None."
- "sleep apnea" category the median age appears to be around 50 year old .
- For the "Insomnia" category, the box plot shows a relatively narrow box, indicating a smaller range of age values approximetly between (42_46).
- The "None" category exhibits a slightly wider range of age. The median age is around 38 years old, . No outliers are visible in this category.
- The box plot suggests that there are differences in Sleep Disorder prevalence across different age groups.

  # Conclusion :
  However, it is important to note that this representation is based on a sample dataset, and individual variations and other factors may influence Sleep 
  Disorder . Further analysis and a larger dataset would be necessary to draw more conclusive insights about the relationship between Sleep Disorder category and age.























   




