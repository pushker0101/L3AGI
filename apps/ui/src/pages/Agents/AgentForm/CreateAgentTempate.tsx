import Typography from 'share-ui/components/typography/Typography'

import {
  StyledHeaderGroup,
  // StyledSectionDescription,
  StyledSectionTitle,
  StyledSectionWrapper,
} from 'pages/Home/homeStyle.css'
import ComponentsWrapper from 'components/ComponentsWrapper/ComponentsWrapper'
import styled from 'styled-components'
import BackButton from 'components/BackButton'

import AgentCard from '../AgentCard'
import { useNavigate } from 'react-router-dom'

import { StyledButtonWrapper } from './CreateAgentForm'
import CreateAgentButtonCard from './components/CreateAgentButtonCard'
import { useModal } from 'hooks'
import { useAgentTemplatesService } from 'services/discover/useAgentTemplatesService'
import TypographyPrimary from 'components/Typography/Primary'
import { useCreateAgent } from '../useCreateAgent'
import { t } from 'i18next'

const CreateAgentTemplate = () => {
  const { refetchAgent } = useCreateAgent()

  const { data: agentsData } = useAgentTemplatesService()

  const { openModal } = useModal()

  const navigate = useNavigate()

  return (
    <StyledSectionWrapper>
      <StyledHeaderGroup className='header_group'>
        <div>
          <StyledSectionTitle>{`${t('add-agent')}`}</StyledSectionTitle>
          {/* <StyledSectionDescription>
            Here are all your agents, managing tasks and operations.
          </StyledSectionDescription> */}
        </div>

        <StyledButtonWrapper>
          <BackButton />
        </StyledButtonWrapper>
      </StyledHeaderGroup>

      <ComponentsWrapper noPadding>
        <StyledTemplatesWrapper>
          <StyledTemplateHeader>
            <CreateAgentButtonCard onClick={() => navigate('/agents/create-agent')} />

            {agentsData?.length > 0 && (
              <TypographyPrimary
                value={t('choose-template')}
                type={Typography.types.LABEL}
                size={Typography.sizes.lg}
              />
            )}
          </StyledTemplateHeader>
          <StyledCardsWrapper>
            {agentsData?.map((agentObj: any, index: number) => {
              const { agent } = agentObj

              return (
                <AgentCard
                  key={index}
                  name={agent.name}
                  description={agent.description}
                  onViewClick={() =>
                    openModal({ name: 'agent-view-modal', data: { agent: agentObj } })
                  }
                  headerTag={agent.role}
                  onCreateClick={async () => {
                    await refetchAgent({ id: agent.id })
                    navigate(`/agents/create-agent?agentId=${agent.id}`)
                  }}
                  // creator={agent.creator}
                  avatar={agent.avatar}
                />
              )
            })}
          </StyledCardsWrapper>
        </StyledTemplatesWrapper>
      </ComponentsWrapper>
    </StyledSectionWrapper>
  )
}

export default CreateAgentTemplate

const StyledTemplatesWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;

  height: calc(100vh - 225px);

  max-height: 1000px;
  overflow-y: auto;
  padding: 0 20px;
  width: 100%;
  /* max-width: 600px; */
`
const StyledTemplateHeader = styled.div`
  display: flex;
  flex-direction: column;
  gap: 26px;
`
const StyledCardsWrapper = styled.div`
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
`
