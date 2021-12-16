#include "SVKElasticForcefield_SOFA_Hexa.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

SVKElasticForcefield_SOFA_Hexa::SVKElasticForcefield_SOFA_Hexa()
: d_youngModulus(initData(&d_youngModulus,
                          Real(1000), "youngModulus",
                          "Young's modulus of the material",
                          true /*displayed_in_GUI*/, false /*read_only_in_GUI*/))
, d_poissonRatio(initData(&d_poissonRatio,
                          Real(0.3),  "poissonRatio",
                          "Poisson's ratio of the material",
                          true /*displayed_in_GUI*/, false /*read_only_in_GUI*/))
, d_topology_container(initLink(
        "topology", "Topology that contains the elements on which this force will be computed."))
{
}

void SVKElasticForcefield_SOFA_Hexa::init() {
    using Mat33 = Eigen::Matrix<double, 3, 3>;
    ForceField::init();

    if (!this->mstate.get() || !d_topology_container.get()) {
        msg_error() << "Both a mechanical object and a topology container are required";
    }

    auto * state = this->mstate.get();
    auto * topology = d_topology_container.get();

    // Convert SOFA position vector to an Eigen matrix (nx3 for n nodes)
    const auto sofa_x0 = state->readRestPositions();
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  x0 (sofa_x0.ref().data()->data(),  state->getSize(), 3);

    // Convert the node index vector from SOFA to an Eigen matrix (nxm for n elements of m nodes each)
    Eigen::Map<const Eigen::Matrix<sofa::Index, Eigen::Dynamic, Element::NumberOfNodes, Eigen::RowMajor>> node_indices (
        topology->getHexas().data()->data(), topology->getNbHexahedra(), Element::NumberOfNodes
    );

    // Gather the integration points for each hexahedron
    const auto number_of_elements = topology->getNbHexahedra();
    p_gauss_nodes.resize(number_of_elements);

    for (Eigen::Index element_id = 0; element_id < number_of_elements; ++element_id) {

        // Position vector of each of the element nodes
        Eigen::Matrix<double, Element::NumberOfNodes, 3, Eigen::RowMajor> node_positions;
        for (Eigen::Index node_id = 0; node_id < Element::NumberOfNodes; ++node_id) {
            node_positions.row(node_id) = x0.row(node_indices(element_id, node_id));
        }

        // Initialize each Gauss nodes
        for (std::size_t gauss_node_id = 0; gauss_node_id < Element::NumberOfGaussNode; ++ gauss_node_id) {
            const auto & gauss_node = Element::gauss_nodes()[gauss_node_id];
            const auto dN_du = Element::dN(gauss_node.position[0], gauss_node.position[1], gauss_node.position[2]);
            const Mat33 J = node_positions.transpose() * dN_du;
            const Mat33 Jinv = J.inverse();
            const auto detJ = J.determinant();
            const Eigen::Matrix<double, Element::NumberOfNodes, 3> dN_dx = (Jinv.transpose() * dN_du.transpose()).transpose();

            p_gauss_nodes[element_id][gauss_node_id].weight = gauss_node.weight;
            p_gauss_nodes[element_id][gauss_node_id].jacobian_determinant = detJ;
            p_gauss_nodes[element_id][gauss_node_id].dN_dx = dN_dx;
        }
    }
}

double SVKElasticForcefield_SOFA_Hexa::getPotentialEnergy(const sofa::core::MechanicalParams *,
                                                              const Data<sofa::type::vector<Coord>> & d_x) const {
    using Mat33 = Eigen::Matrix<double, 3, 3>;

    if (!this->mstate.get() || !d_topology_container.get()) {
        return 0;
    }

    auto * state = this->mstate.get();
    auto * topology = d_topology_container.get();
    const auto  poisson_ratio = d_poissonRatio.getValue();
    const auto  young_modulus = d_youngModulus.getValue();
    const auto mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
    const auto lm = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    static const auto Id = Mat33::Identity();


    // Convert SOFA input position vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_x = sofa::helper::getReadAccessor(d_x);
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  x (sofa_x.ref().data()->data(),  state->getSize(), 3);

    // Convert the node index vector from SOFA to an Eigen matrix (nxm for n elements of m nodes each)
    Eigen::Map<const Eigen::Matrix<sofa::Index, Eigen::Dynamic, Element::NumberOfNodes, Eigen::RowMajor>> node_indices (
            topology->getHexas().data()->data(), topology->getNbHexahedra(), Element::NumberOfNodes
    );

    double Psi = 0.;

    const auto nb_elements = topology->getNbHexahedra();
    for (Eigen::Index element_id = 0; element_id < nb_elements; ++element_id) {
        // Position vector of each of the element nodes
        Eigen::Matrix<double, Element::NumberOfNodes, 3, Eigen::RowMajor> node_positions;
        for (Eigen::Index node_id = 0; node_id < Element::NumberOfNodes; ++node_id) {
            node_positions.row(node_id) = x.row(node_indices(element_id, node_id));
        }

        for (const GaussNode & gauss_node : p_gauss_nodes[element_id]) {
            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto & detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto & dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto & w = gauss_node.weight;

            // Deformation tensor at gauss node
            const Mat33 F = node_positions.transpose()*dN_dx;

            // Green-Lagrange strain tensor at gauss node
            const Mat33 E = 1/2. * (F.transpose() * F - Id);
            const double trE  = E.trace();
            const double trEE = (E*E).trace();

            // Add the potential energy at gauss node
            Psi += (detJ * w) * lm/2.*(trE*trE) + mu*trEE;
        }
    }

    return Psi;
}

void SVKElasticForcefield_SOFA_Hexa::addForce(const sofa::core::MechanicalParams */*mparams*/,
                                                  Data<sofa::type::vector<Deriv>> &d_f,
                                                  const Data<sofa::type::vector<Coord>> &d_x,
                                                  const Data<sofa::type::vector<Deriv>> &/*d_v*/) {
    using Mat33 = Eigen::Matrix<double, 3, 3>;
    using Vec3  = Eigen::Matrix<double, 3, 1>;

    if (!this->mstate.get() || !d_topology_container.get()) {
        return;
    }

    auto * state = this->mstate.get();
    auto * topology = d_topology_container.get();
    const auto  poisson_ratio = d_poissonRatio.getValue();
    const auto  young_modulus = d_youngModulus.getValue();
    const auto mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
    const auto lm = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    static const auto Id = Mat33::Identity();


    // Convert SOFA input position vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_x = sofa::helper::getReadAccessor(d_x);
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  x (sofa_x.ref().data()->data(),  state->getSize(), 3);

    // Convert SOFA output residual vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_f = sofa::helper::getWriteAccessor(d_f);
    Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  R (&(sofa_f[0][0]),  state->getSize(), 3);

    // Convert the node index vector from SOFA to an Eigen matrix (nxm for n elements of m nodes each)
    Eigen::Map<const Eigen::Matrix<sofa::Index, Eigen::Dynamic, Element::NumberOfNodes, Eigen::RowMajor>> node_indices (
            topology->getHexas().data()->data(), topology->getNbHexahedra(), Element::NumberOfNodes
    );

    // Assemble the residual vector
    const auto nb_elements = topology->getNbHexahedra();
    for (Eigen::Index element_id = 0; element_id < nb_elements; ++element_id) {

        // Position vector of each of the element nodes
        Eigen::Matrix<double, Element::NumberOfNodes, 3, Eigen::RowMajor> node_positions;
        for (Eigen::Index node_id = 0; node_id < Element::NumberOfNodes; ++node_id) {
            node_positions.row(node_id) = x.row(node_indices(element_id, node_id));
            if (element_id==0) std::cout<<x.row(node_indices(element_id, node_id)) << "\n";
        }

        for (const GaussNode & gauss_node : p_gauss_nodes[element_id]) {

            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto & detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto & dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto & w = gauss_node.weight;

            // Deformation tensor at gauss node
            const Mat33 F = node_positions.transpose()*dN_dx;

            // Green-Lagrange strain tensor at gauss node
            const Mat33 E = 1/2. * (F.transpose() * F - Id);

            // Second Piola-Kirchhoff stress tensor at gauss node
            const Mat33 S = lm*E.trace()*Id + 2*mu*E;

            // Elastic forces (residual) w.r.t the gauss node applied on each nodes
            for (Eigen::Index i = 0; i < Element::NumberOfNodes; ++i) {
                const auto dx = dN_dx.row(i).transpose();
                const Vec3 f_ = (detJ * w) * F*S*dx;
//                if (element_id == 0) std::cout << f_.transpose() << "\n";
                R.row(node_indices(element_id, i)).noalias() -= f_.transpose();
            }
        }
    }
}

void SVKElasticForcefield_SOFA_Hexa::addKToMatrix(sofa::defaulttype::BaseMatrix * matrix,
                                                      double kFact,
                                                      unsigned int & offset) {
    using Mat33 = Eigen::Matrix<double, 3, 3>;
    using Vec3  = Eigen::Matrix<double, 3, 1>;

    if (!this->mstate.get() || !d_topology_container.get()) {
        return;
    }

    auto * state = this->mstate.get();
    auto * topology = d_topology_container.get();
    const auto  poisson_ratio = d_poissonRatio.getValue();
    const auto  young_modulus = d_youngModulus.getValue();
    const auto mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
    const auto lm = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    static const auto Id = Mat33::Identity();


    // Convert SOFA input position vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_x = state->readPositions();
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  x (sofa_x.ref().data()->data(),  state->getSize(), 3);

    // Convert the node index vector from SOFA to an Eigen matrix (nxm for n elements of m nodes each)
    Eigen::Map<const Eigen::Matrix<sofa::Index, Eigen::Dynamic, Element::NumberOfNodes, Eigen::RowMajor>> node_indices (
            topology->getHexas().data()->data(), topology->getNbHexahedra(), Element::NumberOfNodes
    );

    // Assemble the stiffness matrix
    const auto nb_elements = topology->getNbHexahedra();
    for (Eigen::Index element_id = 0; element_id < nb_elements; ++element_id) {

        // Position vector of each of the element nodes
        Eigen::Matrix<double, Element::NumberOfNodes, 3, Eigen::RowMajor> node_positions;
        for (Eigen::Index node_id = 0; node_id < Element::NumberOfNodes; ++node_id) {
            node_positions.row(node_id) = x.row(node_indices(element_id, node_id));
        }

        for (const GaussNode & gauss_node : p_gauss_nodes[element_id]) {

            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto & detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto & dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto & w = gauss_node.weight;

            // Deformation tensor at gauss node
            // (Note: this is usually saved up for each Gauss node at the residual assembly step to improve a bit the computational cost)
            const Mat33 F = node_positions.transpose()*dN_dx;

            // Green-Lagrange strain tensor at gauss node
            const Mat33 E = 1/2. * (F.transpose() * F - Id);

            // Second Piola-Kirchhoff stress tensor at gauss node
            const Mat33 S = lm*E.trace()*Id + 2*mu*E;

            Eigen::Matrix<double, 6, 6> D;
            D <<
                   lm + 2*mu,  lm,         lm,       0,  0,  0,
                   lm,      lm + 2*mu,     lm,       0,  0,  0,
                   lm,         lm,       lm + 2*mu,  0,  0,  0,
                    0,          0,          0,      mu,  0,  0,
                    0,          0,          0,       0, mu,  0,
                    0,          0,          0,       0,  0, mu;

            // Symmetric elemental tangent stiffness matrix
            for (Eigen::Index i = 0; i < Element::NumberOfNodes; ++i) {
                const Vec3 dxi = dN_dx.row(i).transpose();
                const auto I   = static_cast<int>(offset + node_indices(element_id, i) * 3);
                Eigen::Matrix<double, 6,3> Bi;
                Bi <<
                        F(0,0)*dxi[0],                 F(1,0)*dxi[0],                 F(2,0)*dxi[0],
                        F(0,1)*dxi[1],                 F(1,1)*dxi[1],                 F(2,1)*dxi[1],
                        F(0,2)*dxi[2],                 F(1,2)*dxi[2],                 F(2,2)*dxi[2],
                        F(0,0)*dxi[1] + F(0,1)*dxi[0], F(1,0)*dxi[1] + F(1,1)*dxi[0], F(2,0)*dxi[1] + F(2,1)*dxi[0],
                        F(0,1)*dxi[2] + F(0,2)*dxi[1], F(1,1)*dxi[2] + F(1,2)*dxi[1], F(2,1)*dxi[2] + F(2,2)*dxi[1],
                        F(0,0)*dxi[2] + F(0,2)*dxi[0], F(1,0)*dxi[2] + F(1,2)*dxi[0], F(2,0)*dxi[2] + F(2,2)*dxi[0];

                // The 3x3 sub-matrix Kii is symmetric
                Mat33 Kii = (dxi.dot(S*dxi)*Id + Bi.transpose()*D*Bi) * (detJ * w) * -kFact;
//                if (element_id==0) std::cout << Kii;
                for (int m = 0; m < 3; ++m) {
                    matrix->add(I + m, I + m, Kii(m, m));
                    for (int n = m+1; n < 3; ++n) {
                        matrix->add(I + m, I + n, Kii(m, n));
                        matrix->add(I + n, I + m, Kii(m, n));
                    }
                }

                // We now loop only on the upper triangular part of the
                // element stiffness matrix Ke since it is symmetric
                for (Eigen::Index j = i+1; j < Element::NumberOfNodes; ++j) {
                    const Vec3 dxj = dN_dx.row(j).transpose();
                    const auto J   = static_cast<int>(offset + node_indices(element_id,j) * 3);
                    Eigen::Matrix<double, 6,3> Bj;
                    Bj <<
                            F(0,0)*dxj[0],                 F(1,0)*dxj[0],                 F(2,0)*dxj[0],
                            F(0,1)*dxj[1],                 F(1,1)*dxj[1],                 F(2,1)*dxj[1],
                            F(0,2)*dxj[2],                 F(1,2)*dxj[2],                 F(2,2)*dxj[2],
                            F(0,0)*dxj[1] + F(0,1)*dxj[0], F(1,0)*dxj[1] + F(1,1)*dxj[0], F(2,0)*dxj[1] + F(2,1)*dxj[0],
                            F(0,1)*dxj[2] + F(0,2)*dxj[1], F(1,1)*dxj[2] + F(1,2)*dxj[1], F(2,1)*dxj[2] + F(2,2)*dxj[1],
                            F(0,0)*dxj[2] + F(0,2)*dxj[0], F(1,0)*dxj[2] + F(1,2)*dxj[0], F(2,0)*dxj[2] + F(2,2)*dxj[0];

                    // The 3x3 sub-matrix Kij is NOT symmetric, we store its full part
                    const Mat33 Kij = (dxi.dot(S*dxj)*Id + Bi.transpose()*D*Bj) * (detJ * w) * -kFact;
//                    if (element_id==0) std::cout << Kij;
                    for (int m = 0; m < 3; ++m) {
                        for (int n = 0; n < 3; ++n) {
                            matrix->add(I + m, J + n, Kij(m, n));
                            matrix->add(J + n, I + m, Kij(m, n));
                        }
                    }
                }
            }
        }
    }
    for (int i =0; i<25; i++) {
        for (int j =0; j<25 ;j++ ) {
                std :: cout << matrix->element(i,j) << "\t";
                if (j==24) std::cout << "\n";
        }

    }
}

void SVKElasticForcefield_SOFA_Hexa::addDForce(const sofa::core::MechanicalParams * /*mparams*/,
                                     SVKElasticForcefield_SOFA_Hexa::Data<sofa::type::vector<sofa::type::Vec3>> & /*d_df*/,
                                     const SVKElasticForcefield_SOFA_Hexa::Data<sofa::type::vector<sofa::type::Vec3>> & /*d_dx*/) {
    // Here you would compute df = K*dx
}

void SVKElasticForcefield_SOFA_Hexa::draw(const sofa::core::visual::VisualParams *vparams) {
    auto * topology = d_topology_container.get();
    if (!topology)
        return;

    if (!vparams->displayFlags().getShowForceFields())
        return;

    vparams->drawTool()->saveLastState();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,true);

    vparams->drawTool()->disableLighting();

    const VecCoord& x = this->mstate->read(sofa::core::ConstVecCoordId::position())->getValue();

    std::vector< sofa::type::Vector3 > points[6];
    const auto number_of_elements = topology->getNbHexahedra();
    for (std::size_t hexa_id = 0; hexa_id < number_of_elements; ++hexa_id) {
        const auto & node_indices = topology->getHexahedron(static_cast<sofa::Index>(hexa_id));

        auto a = node_indices[0];
        auto b = node_indices[1];
        auto d = node_indices[3];
        auto c = node_indices[2];
        auto e = node_indices[4];
        auto f = node_indices[5];
        auto h = node_indices[7];
        auto g = node_indices[6];


        Coord center = (x[a]+x[b]+x[c]+x[d]+x[e]+x[g]+x[f]+x[h])*0.125;
        Real percentage = 0.15;
        Coord pa = x[a]-(x[a]-center)*percentage;
        Coord pb = x[b]-(x[b]-center)*percentage;
        Coord pc = x[c]-(x[c]-center)*percentage;
        Coord pd = x[d]-(x[d]-center)*percentage;
        Coord pe = x[e]-(x[e]-center)*percentage;
        Coord pf = x[f]-(x[f]-center)*percentage;
        Coord pg = x[g]-(x[g]-center)*percentage;
        Coord ph = x[h]-(x[h]-center)*percentage;



        points[0].push_back(pa);
        points[0].push_back(pb);
        points[0].push_back(pc);
        points[0].push_back(pa);
        points[0].push_back(pc);
        points[0].push_back(pd);

        points[1].push_back(pe);
        points[1].push_back(pf);
        points[1].push_back(pg);
        points[1].push_back(pe);
        points[1].push_back(pg);
        points[1].push_back(ph);

        points[2].push_back(pc);
        points[2].push_back(pd);
        points[2].push_back(ph);
        points[2].push_back(pc);
        points[2].push_back(ph);
        points[2].push_back(pg);

        points[3].push_back(pa);
        points[3].push_back(pb);
        points[3].push_back(pf);
        points[3].push_back(pa);
        points[3].push_back(pf);
        points[3].push_back(pe);

        points[4].push_back(pa);
        points[4].push_back(pd);
        points[4].push_back(ph);
        points[4].push_back(pa);
        points[4].push_back(ph);
        points[4].push_back(pe);

        points[5].push_back(pb);
        points[5].push_back(pc);
        points[5].push_back(pg);
        points[5].push_back(pb);
        points[5].push_back(pg);
        points[5].push_back(pf);
    }

    vparams->drawTool()->drawTriangles(points[0], sofa::type::RGBAColor(0.7f,0.7f,0.1f,1.0f));
    vparams->drawTool()->drawTriangles(points[1], sofa::type::RGBAColor(0.7f,0.0f,0.0f,1.0f));
    vparams->drawTool()->drawTriangles(points[2], sofa::type::RGBAColor(0.0f,0.7f,0.0f,1.0f));
    vparams->drawTool()->drawTriangles(points[3], sofa::type::RGBAColor(0.0f,0.0f,0.7f,1.0f));
    vparams->drawTool()->drawTriangles(points[4], sofa::type::RGBAColor(0.1f,0.7f,0.7f,1.0f));
    vparams->drawTool()->drawTriangles(points[5], sofa::type::RGBAColor(0.7f,0.1f,0.7f,1.0f));


    std::vector< sofa::type::Vector3 > ignored_points[6];
    for (std::size_t hexa_id = 0; hexa_id < number_of_elements; ++hexa_id) {
        const auto & node_indices = topology->getHexahedron(static_cast<sofa::Index>(hexa_id));

        auto a = node_indices[0];
        auto b = node_indices[1];
        auto d = node_indices[3];
        auto c = node_indices[2];
        auto e = node_indices[4];
        auto f = node_indices[5];
        auto h = node_indices[7];
        auto g = node_indices[6];


        Coord center = (x[a]+x[b]+x[c]+x[d]+x[e]+x[g]+x[f]+x[h])*0.125;
        Real percentage = 0.15;
        Coord pa = x[a]-(x[a]-center)*percentage;
        Coord pb = x[b]-(x[b]-center)*percentage;
        Coord pc = x[c]-(x[c]-center)*percentage;
        Coord pd = x[d]-(x[d]-center)*percentage;
        Coord pe = x[e]-(x[e]-center)*percentage;
        Coord pf = x[f]-(x[f]-center)*percentage;
        Coord pg = x[g]-(x[g]-center)*percentage;
        Coord ph = x[h]-(x[h]-center)*percentage;



        ignored_points[0].push_back(pa);
        ignored_points[0].push_back(pb);
        ignored_points[0].push_back(pc);
        ignored_points[0].push_back(pa);
        ignored_points[0].push_back(pc);
        ignored_points[0].push_back(pd);

        ignored_points[1].push_back(pe);
        ignored_points[1].push_back(pf);
        ignored_points[1].push_back(pg);
        ignored_points[1].push_back(pe);
        ignored_points[1].push_back(pg);
        ignored_points[1].push_back(ph);

        ignored_points[2].push_back(pc);
        ignored_points[2].push_back(pd);
        ignored_points[2].push_back(ph);
        ignored_points[2].push_back(pc);
        ignored_points[2].push_back(ph);
        ignored_points[2].push_back(pg);

        ignored_points[3].push_back(pa);
        ignored_points[3].push_back(pb);
        ignored_points[3].push_back(pf);
        ignored_points[3].push_back(pa);
        ignored_points[3].push_back(pf);
        ignored_points[3].push_back(pe);

        ignored_points[4].push_back(pa);
        ignored_points[4].push_back(pd);
        ignored_points[4].push_back(ph);
        ignored_points[4].push_back(pa);
        ignored_points[4].push_back(ph);
        ignored_points[4].push_back(pe);

        ignored_points[5].push_back(pb);
        ignored_points[5].push_back(pc);
        ignored_points[5].push_back(pg);
        ignored_points[5].push_back(pb);
        ignored_points[5].push_back(pg);
        ignored_points[5].push_back(pf);
    }

    vparams->drawTool()->drawTriangles(ignored_points[0], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[1], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[2], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[3], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[4], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[5], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));


    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,false);

    vparams->drawTool()->restoreLastState();
}

void SVKElasticForcefield_SOFA_Hexa::computeBBox(const sofa::core::ExecParams * /*params*/, bool onlyVisible) {
    using namespace sofa::core::objectmodel;

    if (!onlyVisible) return;
    if (!this->mstate) return;

    sofa::helper::ReadAccessor<Data < VecCoord>>
            x = this->mstate->read(sofa::core::VecCoordId::position());

    static const Real max_real = std::numeric_limits<Real>::max();
    static const Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real, min_real, min_real};
    Real minBBox[3] = {max_real, max_real, max_real};
    for (size_t i = 0; i < x.size(); i++) {
        for (int c = 0; c < 3; c++) {
            if (x[i][c] > maxBBox[c]) maxBBox[c] = static_cast<Real>(x[i][c]);
            else if (x[i][c] < minBBox[c]) minBBox[c] = static_cast<Real>(x[i][c]);
        }
    }

    this->f_bbox.setValue(sofa::type::TBoundingBox<Real>(minBBox, maxBBox));
}


using sofa::core::RegisterObject;
[[maybe_unused]]
static int _c_ = RegisterObject("Simple implementation of a Saint-Venant-Kirchhoff force field for tetrahedral meshes.")
 .add<SVKElasticForcefield_SOFA_Hexa>();
