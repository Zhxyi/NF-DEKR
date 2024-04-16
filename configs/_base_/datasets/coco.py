# huaxi dataset
dataset_info = dict(
    dataset_name='coco_ivd',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='C2_4',
             id=0,
             color=[51, 153, 255],
             type='upper',
             swap=''),
        1:
        dict(
            name='C2_3',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='C3_2',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='C3_1',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        4:
        dict(name='C3_4',
             id=5,
             color=[51, 153, 255],
             type='upper',
             swap=''),
        5:
        dict(
            name='C3_3',
            id=6,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        6:
        dict(
            name='C4_2',
            id=7,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        7:
        dict(
            name='C4_1',
            id=8,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        8:
        dict(name='C4_4',
             id=9,
             color=[51, 153, 255],
             type='lower',
             swap=''),
        9:
        dict(
            name='C4_3',
            id=10,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        10:
        dict(
            name='C5_2',
            id=11,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        11:
        dict(
            name='C5_1',
            id=12,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        12:
        dict(name='C5_4',
             id=13,
             color=[51, 153, 255],
             type='lower',
             swap=''),
        13:
        dict(
            name='C5_3',
            id=14,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        14:
        dict(
            name='C6_2',
            id=15,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        15:
        dict(
            name='C6_1',
            id=16,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        16:
        dict(name='C6_4',
             id=17,
             color=[51, 153, 255],
             type='lower',
             swap=''),
        17:
        dict(
            name='C6_3',
            id=18,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        18:
        dict(
            name='C7_2',
            id=19,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        19:
        dict(
            name='C7_1',
            id=20,
            color=[51, 153, 255],
            type='upper',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('C2_4', 'C3_1'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('C2_3', 'C3_2'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('C3_4', 'C4_1'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('C3_3', 'C4_2'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('C4_4', 'C5_1'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('C4_3', 'C5_2'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('C5_4', 'C6_1'), id=6, color=[0, 255, 0]),
        7:
        dict(link=('C5_3', 'C6_2'), id=7, color=[0, 255, 0]),
        9:
        dict(link=('C6_4', 'C7_1'), id=8, color=[0, 255, 0]),
        10:
        dict(link=('C6_3', 'C7_2'), id=9, color=[0, 255, 0]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.025, 0.026, 0.025, 0.025, 0.025, 0.026, 0.025, 0.025, 0.025, 0.026, 0.025, 0.025, 0.025,
        0.026, 0.025, 0.025, 0.025
    ])

# public dataset
# dataset_info = dict(
#     dataset_name='coco_ivd',
#     paper_info=dict(
#         author='Lin, Tsung-Yi and Maire, Michael and '
#         'Belongie, Serge and Hays, James and '
#         'Perona, Pietro and Ramanan, Deva and '
#         r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
#         title='Microsoft coco: Common objects in context',
#         container='European conference on computer vision',
#         year='2014',
#         homepage='http://cocodataset.org/',
#     ),
#     keypoint_info={
#         0:
#         dict(name='C1_center', id=0, color=[51, 153, 255], type='upper', swap=''),
#         1:
#         dict(
#             name='C2_center',
#             id=1,
#             color=[51, 153, 255],
#             type='upper',
#             swap=''),
#         2:
#         dict(
#             name='C3_center',
#             id=2,
#             color=[51, 153, 255],
#             type='upper',
#             swap=''),
#         3:
#         dict(
#             name='C4_center',
#             id=3,
#             color=[51, 153, 255],
#             type='upper',
#             swap=''),
#         4:
#         dict(
#             name='C5_center',
#             id=4,
#             color=[51, 153, 255],
#             type='upper',
#             swap=''),
#         5:
#         dict(
#             name='C6_center',
#             id=5,
#             color=[0, 255, 0],
#             type='lower',
#             swap=''),
#         6:
#         dict(
#             name='C7_center',
#             id=6,
#             color=[255, 128, 0],
#             type='lower',
#             swap=''),
#         7:
#         dict(
#             name='T1_center',
#             id=7,
#             color=[0, 255, 0],
#             type='lower',
#             swap=''),
#         8:
#         dict(
#             name='T2_center',
#             id=8,
#             color=[255, 128, 0],
#             type='lower',
#             swap=''),
#     },
#     skeleton_info={
#         0:
#         dict(link=('C1_center', 'C2_center'), id=0, color=[0, 255, 0]),
#         1:
#         dict(link=('C2_center', 'C3_center'), id=1, color=[0, 255, 0]),
#         2:
#         dict(link=('C3_center', 'C4_center'), id=2, color=[255, 128, 0]),
#         3:
#         dict(link=('C4_center', 'C5_center'), id=3, color=[255, 128, 0]),
#         4:
#         dict(link=('C5_center', 'C6_center'), id=4, color=[51, 153, 255]),
#         5:
#         dict(link=('C6_center', 'C7_center'), id=5, color=[51, 153, 255]),
#         6:
#         dict(link=('C7_center', 'T1_center'), id=6, color=[51, 153, 255]),
#         7:
#         dict(
#             link=('T1_center', 'T2_center'),
#             id=7,
#             color=[51, 153, 255]),
#     },
#     joint_weights=[
#         1., 1., 1., 1., 1., 1., 1., 1.2, 1.2
#     ],
#     sigmas=[
#         0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072
#     ])