from data_preprocessing import data
import seaborn as sns
import matplotlib.pyplot as plt

#Example Plot:
sns.countplot(x = 'stroke' , data = data)
plt.show()