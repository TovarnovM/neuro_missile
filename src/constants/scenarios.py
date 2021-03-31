import numpy as np
import matplotlib.pyplot as plt

scenarios = {
  'scenario_Chengdu_J_10': [
    { 
      "trg_pos_0": (1450, 3200),
      "trg_vels": [
        (0,	[150,	90]),
        (2,	[170,	50]),
        (4,	[130,	-80]),
        (6,	[90, -120]),
        (8,	[90,	-150]),
        (10, [80, -170]),
        (12, [70, -170]),
        (14, [60, -170]),
        (16, [70, -130]),
        (18, [80, -90]),
        (20, [90, -60]),
        (22, [120, -20]),
        (24, [130, 10]),
        (26, [130, 80]),
        (28, [130, 120]),
        (30, [130, 120]),
        (32, [130, 130]),
        (34, [150, 100]),
        (36, [170, 80]),
        (38, [175, 60]),
        (40, [180, 40]),
        (42, [185, 20]),
        (44, [186, 0]),
        (46, [186, -20]),
        (48, [170, -40]),
        (50, [160, -60]),
        (52, [160, -80]),
        (54, [160, -100])
      ]
    },{
      "trg_pos_0": (-450, 4200),
      "trg_vels": [
        (0,	[-130, -130]),
        (2,	[-130, -130]),
        (4,	[-130, -130]),
        (6,	[-130, -130]),
        (8,	[-130, -130]),
        (10,	[-130, -130]),
        (12,	[-130, -130]),
        (14,	[-130, -130]),
        (16,	[-130, -130]),
        (18,	[-130, -133.3333333]),
        (20,	[-140, -118.3333333]),
        (22,	[-150, -103.3333333]),
        (24,	[-160, -88.33333333]),
        (26,	[-170, -73.33333333]),
        (28,	[-175, -58.33333333]),
        (30,	[-180, -43.33333333]),
        (32,	[-185, -30]),
        (34,	[-185, -10]),
        (36,	[-185, 0]),
        (38,	[-170, -50]),
        (40,	[-160, -100]),
        (42,	[-170, -50]),
        (44,	[-180, 0]),
        (46,	[-170, -50]),
        (48,	[-170, -50]),
        (50,	[-170, -70]),
        (52,	[-170, -10]),
        (54,	[-170, 50])
      ]
    }
  ],
  'scenario_Refale': [
    { 
      "trg_pos_0": (0, 7900),
      "trg_vels": [
        (0,	[60,	-529]),
        (2,	[60,	-510]),
        (4,	[60,	-491]),
        (6,	[20,	-472]),
        (8,	[0,	-453]),
        (10,	[-10, -434]),
        (12,	[-90, -415]),
        (14,	[-200, -492]),
        (16,	[-350, -377]),
        (18,	[-440, -158]),
        (20,	[-500, -39]),
        (22,	[-400, 80]),
        (24,	[-300, 199]),
        (26,	[-200, 318]),
        (28,	[-100, 437]),
        (30,	[0,	406]),
        (32,	[100,	475]),
        (34,	[200,	444]),
        (36,	[300,	413]),
        (38,	[370,	382]),
        (40,	[398,	351]),
        (42,	[426,	320]),
        (44,	[444,	289]),
        (46,	[444,	258]),
        (48,	[444,	227]),
        (50,	[444,	196]),
        (52,	[444,	165]),
        (54,	[444,	134])
        ]
    },{ 
      "trg_pos_0": (-15150, 990),
      "trg_vels": [
        (0,	[490,	-130]),
        (2,	[490,	-110]),
        (4,	[490,	-60]),
        (6,	[490,	0]),
        (8,	[490,	60]),
        (10,	[490,	0]),
        (12,	[490,	-60]),
        (14,	[490,	0]),
        (16,	[490,	60]),
        (18,	[490,	0]),
        (20,	[490,	-60]),
        (22,	[490,	-60]),
        (24,	[430,	22]),
        (26,	[430,	42]),
        (28,	[430,	102]),
        (30,	[430,	232]),
        (32,	[430,	212]),
        (34,	[430,	302]),
        (36,	[430,	212]),
        (38,	[430,	242]),
        (40,	[430,	242]),
        (42,	[430,	242]),
        (44,	[430,	242]),
        (46,	[430,	242]),
        (48,	[430,	242]),
        (50,	[430,	242]),
        (52,	[430,	242]),
        (54,	[430,	242])
      ]
    }
  ],
  'scenario_RaptorF-22': [
    {
      "trg_pos_0": (-3450, 320),
      "trg_vels": [
        (0,	[560,	230]),
        (2,	[560,	320]),
        (4,	[340,	450]),
        (6,	[150,	560]),
        (8,	[13,	630]),
        (10,	[0,	665]),
        (12,	[130,	630]),
        (14,	[0,	620]),
        (16,	[130,	630]),
        (18,	[220,	540]),
        (20,	[340,	400]),
        (22,	[460,	260]),
        (24,	[580,	120]),
        (26,	[580,	-20]),
        (28,	[580,	-160]),
        (30,	[580,	-300])
      ]
    },{
      "trg_pos_0": (-4000, 5900),
      "trg_vels": [
        (0,	[530,	-140]),
        (2,	[500,	-220]),
        (4,	[450,	-200]),
        (6,	[400,	-100]),
        (8,	[350,	-40]),
        (10,	[300,	40]),
        (12,	[250,	100]),
        (14,	[200,	150]),
        (16,	[150,	230]),
        (18,	[100,	360]),
        (20,	[50,	425]),
        (22,	[0,	425]),
        (24,	[-50,	425]),
        (26,	[-100,	425]),
        (28,	[-150,	425]),
        (30,	[-200,	425]),
        (32,	[-250,	425]),
        (34,	[-300,	425]),
        (36,	[-350,	425]),
        (38,	[-400,	425]),
        (40,	[-450,	425]),
        (42,	[-475,	450])
      ]
    }
  ],
  'scenario_Cy-57': [
    {
      "trg_pos_0": (4000, 5900),
      "trg_vels": [
        (0,	[-509,	0]),
        (2,	[-515,	-130]),
        (4,	[-521,	-250]),
        (6,	[-627,	-130]),
        (8,	[-633,	-100]),
        (10, [-633,	0]),
        (12, [-633,	50]),
        (14, [-633,	0]),
        (16, [-633,	0]),
        (18, [-633,	50]),
        (20, [-633,	140]),
        (22, [-633,	230]),
        (24, [-633,	320]),
        (26, [-633,	410]),
        (28, [-633,	500]),
        (30, [-633,	500]),
        (32, [-633,	500]),
        (34, [-633,	500]),
        (36, [-633,	500]),
        (38, [-633,	500]),
        (40, [-633,	500]),
        (42, [-633,	500]),
        (44, [-633,	500]),
        (46, [-633,	500]),
        (48, [-633,	500]),
        (50, [-633,	500]),
        (52, [-633,	500]),
        (54, [-633,	500])
      ]
    }
  ],
  'SUCCESS': [
      {
        "trg_pos_0": (410, 140),
        "trg_vels": [
          (0,[410,1]),
          (3,[450,140]),
          (4,[470,131]),
          (5,[480,120]),
          (6,[480,110]),
          (7,[480,110]),
          (8,[490,111]),
          (9,[490,114]),
          (10,[470,116]),
          (11,[450,119]),
          (12,[430,115]),
          (13,[420,110]),
          (14,[420,-116]),
          (15,[420,-110]),
          (16,[420,-113]),
          (17,[440,-111])
        ]
      }, {
        "trg_pos_0": (1410, 840),
        "trg_vels":  [
            (0,[410,1]),
            (3,[450,140]),
            (4,[470,131]),
            (5,[480,120]),
            (6,[480,110]),
            (7,[480,110]),
            (8,[490,111]),
            (9,[490,114]),
            (10,[470,116]),
            (11,[450,119]),
            (12,[430,115]),
            (13,[420,110]),
            (14,[420,-116]),
            (15,[420,-110]),
            (16,[420,-113]),
            (17,[440,-111])
        ]
      }, {
        "trg_pos_0": (1410, 1420),
        "trg_vels": [
            (0,[420,20]),
            (1,[428,10]),
            (2,[424,0]),
            (3,[426,-80]),
            (4,[420,-120]),
            (5,[420,-120]),
            (6,[420,-140]),
            (7,[425,-161]),
            (8,[420,-144]),
            (9,[425,-160]),
            (10,[425,-190]),
            (11,[425,-150]),
            (12,[420,-100]),
            (13,[425,-160]),
            (14,[420,-100]),
            (15,[420,-130]),
            (16,[420,-100]),
            (17,[420,-100])
        ]
      }, {
        "trg_pos_0": (1330, 102),
        "trg_vels": [
          (0,[330,102]),
          (1,[338,50]),
          (2,[334,100]),
          (3,[336,90]),
          (4,[330,20]),
          (5,[330,0]),
          (6,[330,0]),
          (7,[335,-20]),
          (8,[350,-40]),
          (9,[365,-60]),
          (10,[375,-100]),
          (11,[385,-150]),
          (12,[390,-100]),
          (13,[405,-60]),
          (14,[430,-100]),
          (15,[440,-30]),
          (16,[430,-10])
        ]
      }, {
        "trg_pos_0": (1330, 900),
        "trg_vels": [
            (0,[330,0]),
            (1,[270,-5]),
            (2,[260,-10]),
            (3,[250,-40]),
            (4,[210,-50]),
            (5,[180,-90]),
            (6,[160,-100]),
            (7,[130,-120]),
            (8,[100,-140]),
            (9,[80,-160]),
            (10,[60,-170]),
            (11,[30,-180]),
            (12,[10,-150]),
            (13,[-50,-120]),
            (14,[-90,-100]),
            (15,[-130,-70]),
            (16,[-200,-50]),
            (17,[-230,-30])
        ]
      }, {
        "trg_pos_0": (1310, 620),
        "trg_vels":[
            (0,[310,120]),
            (2,[342,150]),
            (4,[321,100]),
            (6,[329,60]),
            (8,[332,150]),
            (10,[340,200]),
            (12,[-336,210]),
            (14,[-330,203]),
            (16,[-325,250]),
            (18,[-310,288]),
            (20,[330,210])
        ]
      }
  ],
  'FAIL': [
    {
      "trg_pos_0": (1300, 420),
      "trg_vels": [
        (0,[300,120]),
        (2,[320,250]),
        (4,[350,300]),
        (6,[400,360]),
        (8,[432,-335]),
        (10,[430,-330]),
        (12,[450,-330]),
        (14,[445,-250]),
        (16,[445,-170]),
        (18,[440,-100]),
        (20,[440,-160]),
        (22,[445,-200]),
        (24,[445,-220]),
        (26,[440,-220]),
        (28,[455,-240]),
        (30,[460,-270]),
        (32,[440,-300])
      ]
    },
    {
      "trg_pos_0": (1410, -120),
      "trg_vels": [
        (0,[210,220]),
        (1,[212,270]),
        (2,[221,300]),
        (3,[229,325]),
        (4,[232,350]),
        (5,[240,360]),
        (6,[236,350]),
        (7,[230,400]),
        (8,[225,460]),
        (10,[210,500]),
        (20,[230,580])
      ]
    },
    {
      "trg_pos_0": (1410, 120),
      "trg_vels": [
        (0,[310,420]),
        (2,[342,470]),
        (4,[321,400]),
        (6,[329,425]),
        (8,[332,450]),
        (10,[340,460]),
        (12,[336,450]),
        (14,[330,400]),
        (16,[325,460]),
        (18,[310,400]),
        (20,[330,480])
      ]
    }
  ]
}


if __name__ == "__main__":
  tscenario =  list(map(lambda x: x[0], scenarios['scenario_Chengdu_J_10'][0]['trg_vels']))
  vector_velocity = list(map(lambda x: x[1], scenarios['scenario_Chengdu_J_10'][0]['trg_vels']))
  x_by_scenario, y_by_scenario = [scenarios['scenario_Chengdu_J_10'][0]['trg_pos_0'][0]], [scenarios['scenario_Chengdu_J_10'][0]['trg_pos_0'][1]]

    
  vx_by_scenario =  np.array(vector_velocity)[:,:1]
  vy_by_scenario =  np.array(vector_velocity)[:,1:]
  vx = np.reshape(vx_by_scenario, len(vx_by_scenario))
  vy = np.reshape(vy_by_scenario, len(vy_by_scenario))
  v = np.sqrt(vx ** 2 + vy ** 2)  

  for i in range(len(tscenario)):
    if(i == 0):
      continue
    dt = tscenario[i] - tscenario[i-1]
    x = x_by_scenario[ i - 1 ] + vx[i] * dt
    y = y_by_scenario[ i - 1 ] + vy[i] * dt
    x_by_scenario.append(x)
    y_by_scenario.append(y)


  plt.subplot(2,1,1)
  plt.title('Y(X)', loc='left')
  plt.plot(x_by_scenario, y_by_scenario)
  plt.grid()
  

  plt.subplot(2,1,2)
  plt.title('V(t)', loc='left')
  plt.plot(tscenario, v)
  plt.grid()

  plt.show()
