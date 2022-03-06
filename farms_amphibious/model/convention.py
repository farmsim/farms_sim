"""Oscillator naming convention"""

from farms_data.options import Options


class AmphibiousConvention(Options):
    """Amphibious convention"""

    def __init__(self, **kwargs):
        super().__init__()
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.single_osc_body = kwargs.pop('single_osc_body', False)
        self.single_osc_legs = kwargs.pop('single_osc_legs', False)
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')
        self.links_names = kwargs.pop(
            'links_names',
            [link['name'] for link in kwargs['links']]
            if 'links' in kwargs
            else [
                f'link_body_{link_i}'
                for link_i in range(self.n_joints_body + 1)
            ] + [
                f'link_leg_{leg_i}_{"R" if side_i else "L"}_{joint_i}'
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
                f'joint_body_{joint_i}'
                for joint_i in range(self.n_joints_body)
            ] + [
                f'joint_leg_{leg_i}_{"R" if side_i else "L"}_{joint_i}'
                for leg_i in range(self.n_legs//2)
                for side_i in range(2)
                for joint_i in range(self.n_dof_legs)
            ],
        )
        n_joints = self.n_joints()
        assert len(self.joints_names) == n_joints, (
            f'Provided {len(self.joints_names)} names for joints'
            f' but there should be {n_joints}:'
            f'\n{self.joints_names}'
        )
        assert not kwargs, kwargs

    @classmethod
    def from_amphibious_options(cls, animat_options):
        """From morphology"""
        network = animat_options.control.network
        return cls.from_morphology(
            morphology=animat_options.morphology,
            single_osc_body=network.single_osc_body,
            single_osc_legs=network.single_osc_legs,
        )

    @classmethod
    def from_morphology(cls, morphology, **kwargs):
        """From morphology"""
        return cls(
            n_joints_body=morphology['n_joints_body'],
            n_dof_legs=morphology['n_dof_legs'],
            n_legs=morphology['n_legs'],
            links_names=morphology.links_names(),
            joints_names=morphology.joints_names(),
            **kwargs,
        )

    def n_links_body(self):
        """Number of links in body"""
        return self.n_joints_body+1

    def n_states(self):
        """Number of states"""
        n_osc = self.n_osc()
        n_joints = self.n_joints()
        return (
            n_osc  # Phases
            + n_osc  # Amplitudes
            + n_joints  # Joints offsets
        )

    def n_joints(self):
        """Number of joints"""
        return self.n_joints_body + self.n_joints_legs()

    def n_joints_legs(self):
        """Number of joints"""
        return self.n_legs*self.n_dof_legs

    def n_legs_pair(self):
        """Number of legs pairs"""
        return self.n_legs//2

    def n_osc(self):
        """Number of oscillators"""
        return self.n_osc_body() + self.n_osc_legs()

    def n_opbj(self):
        """Number of oscillators per body joint"""
        return 1 if self.single_osc_body else 2

    def n_oplj(self):
        """Number of oscillators per leg joint"""
        return 1 if self.single_osc_legs else 2

    def n_osc_body(self):
        """Number of body oscillators"""
        return self.n_opbj()*(self.n_joints_body)

    def n_osc_legs(self):
        """Number of legs oscillators"""
        return self.n_oplj()*(self.n_joints_legs())

    def body_osc_indices(self, joint_i):
        """Body oscillator indices"""
        n_body_joints = self.n_joints_body
        assert 0 <= joint_i < n_body_joints, (
            f'Joint must be < {n_body_joints}, got {joint_i}'
        )
        index = self.n_opbj()*joint_i
        return list(range(index, index + self.n_opbj()))

    def bodyosc2index(self, joint_i, side=0):
        """body2index"""
        if self.single_osc_body:
            assert side == 0, f'No oscillator side for joint {joint_i}'
        return self.body_osc_indices(joint_i)[side]

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
        assert 0 <= joint_i < n_body_joints, (
            f'Joint must be < {n_body_joints}, got {joint_i}'
        )
        if self.single_osc_body:
            assert side == 0, f'No oscillator side for joint {joint_i}'
        return (
            f'osc_body_{joint_i}'
            if self.single_osc_body
            else f'osc_body_{joint_i}_{"R" if side else "L"}'
        )

    def leg_osc_indices(self, **kwargs):
        """Leg oscillator indices"""
        leg_opj = self.n_oplj()
        if 'index' in kwargs:
            index = (
                self.n_osc_body()
                + leg_opj*(
                    kwargs.pop('index') - self.n_joints_body
                )
            )
        else:
            leg_i, side_i, joint_i = (
                kwargs.pop(key)
                for key in ('leg_i', 'side_i', 'joint_i')
            )
            n_legs = self.n_legs
            n_legs_dof = self.n_dof_legs
            assert 0 <= leg_i < n_legs, f'Leg must be < {n_legs//2}, got {leg_i}'
            assert 0 <= side_i < 2, f'Body side must be < 2, got {side_i}'
            assert 0 <= joint_i < n_legs_dof, f'Joint must be < {n_legs_dof}, got {joint_i}'
            index = (
                self.n_osc_body()
                + leg_i*2*n_legs_dof*leg_opj  # 2 legs
                + side_i*n_legs_dof*leg_opj
                + joint_i*leg_opj
            )
        assert not kwargs, kwargs
        return list(range(index, index + leg_opj))

    def joint2legindices(self, joint_i):
        """Joint index to leg indices"""
        assert self.n_joints_body <= joint_i < self.n_joints(), (
            f'{self.n_joints_body} !<= {joint_i} !< {self.n_joints()}'
        )
        j_i = joint_i - self.n_joints_body
        dof_i = j_i % self.n_dof_legs
        side_i = ((j_i - dof_i) // self.n_dof_legs) % 2
        leg_i = (j_i - side_i*self.n_dof_legs - dof_i) // (4*self.n_dof_legs)
        return (leg_i, side_i, dof_i)

    def osc_indices(self, joint_i):
        """Joint index to oscillator indices"""
        return (
            self.body_osc_indices(joint_i)
            if joint_i < self.n_joints_body
            else self.leg_osc_indices(index=joint_i)
        )

    def legosc2index(self, leg_i, side_i, joint_i, side=0):
        """legosc2index"""
        if self.single_osc_legs:
            assert side == 0, (
                f'No oscillator side for legs ({leg_i}, {side_i}, {joint_i})'
            )
        else:
            assert 0 <= side < 2, f'Oscillator side must be < 2, got {side}'
        return self.leg_osc_indices(
            leg_i=leg_i,
            side_i=side_i,
            joint_i=joint_i,
        )[side]

    def legosc2name(self, leg_i, side_i, joint_i, side=0):
        """legosc2name"""
        n_legs = self.n_legs
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, f'Leg must be < {n_legs//2}, got {leg_i}'
        assert 0 <= side_i < 2, f'Body side must be < 2, got {side_i}'
        assert 0 <= joint_i < n_legs_dof, f'Joint must be < {n_legs_dof}, got {joint_i}'
        assert 0 <= side < 2, f'Oscillator side must be < 2, got {side}'
        if self.single_osc_legs:
            assert side == 0, (
                f'No oscillator side for legs ({leg_i}, {side_i}, {joint_i})'
            )
        return (
            f'osc_leg_{leg_i}_{"R" if side_i else "L"}_{joint_i}'
            if self.single_osc_legs
            else f'osc_leg_{leg_i}_{"R" if side_i else "L"}_{joint_i}_{side}'
        )

    def oscindex2information(self, osc_i):
        """Oscillator index information"""
        information = {}
        n_oscillators = self.n_osc()
        n_body_oscillators = self.n_osc_body()
        assert 0 <= osc_i < n_oscillators, (
            f'Index {osc_i} bigger than number of oscillators ({n_oscillators})'
        )
        information['body'] = osc_i < n_body_oscillators
        if information['body']:
            information['joint_i'] = osc_i if self.single_osc_body else osc_i//2
            information['side'] = 0 if self.single_osc_body else (osc_i % 2)
        else:
            index_i = osc_i - n_body_oscillators
            information['side'] = 0 if self.single_osc_legs else (index_i % 2)
            n_osc_leg = self.n_oplj()*self.n_dof_legs
            n_osc_leg_pair = 2*n_osc_leg
            information['leg'] = index_i // n_osc_leg
            information['leg_i'] = index_i // n_osc_leg_pair
            information['side_i'] = (
                0 if (index_i % n_osc_leg_pair) < n_osc_leg else 1
            )
            information['joint_i'] = (
                (index_i % n_osc_leg) // self.n_oplj()
            )
        return information

    def bodylink2name(self, link_i):
        """bodylink2name"""
        n_body = self.n_joints_body + 1
        assert 0 <= link_i < n_body, f'Body must be < {n_body}, got {link_i}'
        return self.links_names[link_i]

    def body_links_names(self):
        """Body links names"""
        return [
            self.bodylink2name(link_i)
            for link_i in range(self.n_joints_body + 1)
        ]

    def leglink2index(self, leg_i, side_i, joint_i):
        """leglink2index"""
        n_legs = self.n_legs//2
        n_body_links = self.n_joints_body + 1
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, f'Leg must be < {n_legs//2}, got {leg_i}'
        assert 0 <= side_i < 2, f'Body side must be < 2, got {side_i}'
        assert 0 <= joint_i < n_legs_dof, f'Joint must be < {n_legs_dof}, got {joint_i}'
        return (
            n_body_links
            + leg_i*2*n_legs_dof
            + side_i*n_legs_dof
            + joint_i
        )

    def bodyjoint2name(self, link_i):
        """bodyjoint2name"""
        n_body = self.n_joints_body + 1
        assert 0 <= link_i < n_body, f'Body must be < {n_body}, got {link_i}'
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
        assert 0 <= joint_i < n_body, f'Body joint must be < {n_body}, got {joint_i}'
        return joint_i

    def legjoint2index(self, leg_i, side_i, joint_i):
        """legjoint2index"""
        n_body_joints = self.n_joints_body
        n_legs = self.n_legs
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, f'Leg must be < {n_legs//2}, got {leg_i}'
        assert 0 <= side_i < 2, f'Body side must be < 2, got {side_i}'
        assert 0 <= joint_i < n_legs_dof, f'Joint must be < {n_legs_dof}, got {joint_i}'
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
        assert 0 <= leg_i < n_legs, f'Leg must be < {n_legs//2}, got {leg_i}'
        assert 0 <= side_i < 2, f'Body side must be < 2, got {side_i}'
        return 2*leg_i + side_i

    def contactleglink2name(self, leg_i, side_i):
        """Contact leg link name"""
        return self.leglink2name(leg_i, side_i, self.n_dof_legs-1)
