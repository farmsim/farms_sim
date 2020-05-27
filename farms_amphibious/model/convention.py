"""Oscillator naming convention"""

class AmphibiousConvention:
    """Amphibious convention"""

    def __init__(self, **kwargs):
        super(AmphibiousConvention, self).__init__()
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')
        self.links_names = kwargs.pop(
            'links_names',
            [link['name'] for link in kwargs['links']]
            if 'links' in kwargs
            else [
                'link_body_{}'.format(link_i)
                for link_i in range(self.n_joints_body+1)
            ] + [
                'link_leg_{}_{}_{}'.format(leg_i, 'R' if side_i else 'L', joint_i)
                for leg_i in range(self.n_legs//2)
                for side_i in range(2)
                for joint_i in range(self.n_dof_legs)
            ],
        )
        self.joints_names = kwargs.pop(
            'joints_names',
            [joint['name'] for joint in kwargs['joints']]
            if 'joints' in kwargs
            else [
                'joint_body_{}'.format(joint_i)
                for joint_i in range(self.n_joints_body)
            ] + [
                'joint_leg_{}_{}_{}'.format(leg_i, 'R' if side_i else 'L', joint_i)
                for leg_i in range(self.n_legs//2)
                for side_i in range(2)
                for joint_i in range(self.n_dof_legs)
            ],
        )

    def n_joints(self):
        """Number of joints"""
        return self.n_joints_body + self.n_legs*self.n_dof_legs

    def bodyosc2index(self, joint_i, side=0):
        """body2index"""
        n_body_joints = self.n_joints_body
        assert 0 <= joint_i < n_body_joints, 'Joint must be < {}, got {}'.format(
            n_body_joints,
            joint_i
        )
        return 2*joint_i + side

    def oscindex2name(self, index):
        """Oscillator index to parameters"""
        parameters = self.oscindex2information(index)
        body = parameters.pop('body')
        if not body:
            parameters.pop('leg')
        return (
            self.bodyosc2name(**parameters)
            if body
            else self.legosc2name(**parameters)
        )

    def bodyosc2name(self, joint_i, side=0):
        """body2name"""
        n_body_joints = self.n_joints_body
        assert 0 <= joint_i < n_body_joints, 'Joint must be < {}, got {}'.format(
            n_body_joints,
            joint_i
        )
        return 'osc_body_{}_{}'.format(joint_i, 'R' if side else 'L')

    def legosc2index(self, leg_i, side_i, joint_i, side=0):
        """legosc2index"""
        n_legs = self.n_legs
        n_body_joints = self.n_joints_body
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        assert 0 <= joint_i < n_legs_dof, 'Joint must be < {}, got {}'.format(n_legs_dof, joint_i)
        assert 0 <= side < 2, 'Oscillator side must be < 2, got {}'.format(side)
        return (
            2*n_body_joints
            + leg_i*2*n_legs_dof*2  # 2 oscillators, 2 legs
            + side_i*n_legs_dof*2  # 2 oscillators
            + 2*joint_i
            + side
        )

    def legosc2name(self, leg_i, side_i, joint_i, side=0):
        """legosc2name"""
        n_legs = self.n_legs
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        assert 0 <= joint_i < n_legs_dof, 'Joint must be < {}, got {}'.format(n_legs_dof, joint_i)
        assert 0 <= side < 2, 'Oscillator side must be < 2, got {}'.format(side)
        return 'osc_leg_{}_{}_{}_{}'.format(
            leg_i,
            'R' if side_i else 'L',
            joint_i,
            side,
        )

    def oscindex2information(self, index):
        """Oscillator index information"""
        information = {}
        n_joints = self.n_joints_body + self.n_legs*self.n_dof_legs
        n_oscillators = 2*n_joints
        n_body_oscillators = 2*self.n_joints_body
        assert 0 <= index < n_oscillators, (
            'Index {} bigger than number of oscillator (n={})'.format(
                index,
                n_oscillators,
            )
        )
        information['body'] = index < n_body_oscillators
        information['side'] = index % 2
        if information['body']:
            information['joint_i'] = index//2
        else:
            index_i = index - n_body_oscillators
            n_osc_leg = 2*self.n_dof_legs
            n_osc_leg_pair = 2*n_osc_leg
            information['leg'] = index_i // n_osc_leg
            information['leg_i'] = index_i // n_osc_leg_pair
            information['side_i'] = (
                0 if (index_i % n_osc_leg_pair) < n_osc_leg else 1
            )
            information['joint_i'] = (index_i % n_osc_leg)//2
        return information

    def bodylink2name(self, link_i):
        """bodylink2name"""
        n_body = self.n_joints_body+1
        assert 0 <= link_i < n_body, 'Body must be < {}, got {}'.format(n_body, link_i)
        return self.links_names[link_i]

    def body_links_names(self):
        """Body links names"""
        return [
            self.bodylink2name(link_i)
            for link_i in range(self.n_joints_body+1)
        ]

    def leglink2index(self, leg_i, side_i, joint_i):
        """leglink2index"""
        n_legs = self.n_legs
        n_body_links = self.n_joints_body+1
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        assert 0 <= joint_i < n_legs_dof, 'Joint must be < {}, got {}'.format(n_legs_dof, joint_i)
        return (
            n_body_links
            + leg_i*2*n_legs_dof
            + side_i*n_legs_dof
            + joint_i
        )

    def bodyjoint2name(self, link_i):
        """bodyjoint2name"""
        n_body = self.n_joints_body + 1
        assert 0 <= link_i < n_body, 'Body must be < {}, got {}'.format(n_body, link_i)
        return self.joints_names[link_i]

    def leglink2name(self, leg_i, side_i, joint_i):
        """leglink2name"""
        return self.links_names[self.leglink2index(leg_i, side_i, joint_i)]

    def feet_links_names(self):
        """Feet links names"""
        return [
            self.leglink2name(leg_i, side_i, self.n_dof_legs-1)
            for leg_i in range(self.n_legs//2)
            for side_i in range(2)
        ]

    def bodyjoint2index(self, joint_i):
        """bodyjoint2index"""
        n_body = self.n_joints_body
        assert 0 <= joint_i < n_body, 'Body joint must be < {}, got {}'.format(n_body, joint_i)
        return joint_i

    def legjoint2index(self, leg_i, side_i, joint_i):
        """legjoint2index"""
        n_body_joints = self.n_joints_body
        n_legs = self.n_legs
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        assert 0 <= joint_i < n_legs_dof, 'Joint must be < {}, got {}'.format(n_legs_dof, joint_i)
        return (
            n_body_joints
            + leg_i*2*n_legs_dof
            + side_i*n_legs_dof
            + joint_i
        )

    def legjoint2name(self, leg_i, side_i, joint_i):
        """legjoint2name"""
        return self.joints_names[self.legjoint2index(leg_i, side_i, joint_i)]

    def contactleglink2index(self, leg_i, side_i):
        """Contact leg link 2 index"""
        n_legs = self.n_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        return 2*leg_i + side_i

    def contactleglink2name(self, leg_i, side_i):
        """Contact leg link name"""
        return self.leglink2name(leg_i, side_i, self.n_dof_legs-1)
